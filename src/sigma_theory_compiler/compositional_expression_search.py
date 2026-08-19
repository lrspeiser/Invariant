"""Compositional expression search: trillion-scale brute force over a generative grammar.

Every earlier campaign in this repository enumerated families a **human named**: 21 basis
families, five nonlinear models, continued-fraction patterns ``a_n``/``b_n``, hypergeometric
parameter ratios.  Those families exist because classical mathematicians found them fruitful,
so enumerating them rediscovers classical mathematics -- which is exactly what happened, zero
new results across 3.4e9 candidates.  The diagnosis is that the *generator* was curated, not
that the search was too small.

This module replaces the curated family with a **generative grammar**.  Every expression is a
fixed-length reverse-Polish program over a small closed token alphabet, run by a stack machine
that has no idea what any of it means.  Most programs in the space are expressions nobody has
ever written down.  The machine evaluates them by brute force and never understands them.

Grammar (declared exactly; see :data:`TOKEN_NAMES`)::

    terminals  1 2 3 4 5 1/2 1/3 2/3 3/2 k            (10)
    unary      neg recip sqr sqrt exp ln sin cos atan  (9)
    binary     add sub mul div pow                     (5)

Two search modes over two declared ordinal spaces:

``MODE C`` (constant hunt) enumerates length-9 programs over the 23 non-``k`` tokens --
23^9 = 1,801,152,661,463 ordinals -- each evaluating to a single real matched against a target
list.

``MODE F`` (function hunt) enumerates length-8 programs over all 24 tokens -- 24^8 =
110,075,314,176 ordinals -- keeping only those that mention ``k``, and reads each as the term
``t_k`` of a series ``Sum_{k>=1} t_k`` *and* as a sequence whose limit is ``lim t_k``.  This is
where genuinely unnamed formulas live.  Both spaces are enumerated **exhaustively**: the
campaign's 1.911e12 candidates are a complete sweep of the declared grammar, not a slice.

The ordinal codec is mixed-radix little-endian over the mode's alphabet, an injection from
``[0, base^L)`` onto token sequences, with a validity pass that rejects stack underflow,
overflow past the declared depth, and any sequence not terminating with exactly one value.
The valid fraction is **measured**, never estimated.

**The chance-match gate is the heart of this module.**  At 1e12 candidates and a 1e-12 match
tolerance, spurious matches are guaranteed, so a raw match count means nothing.  Around each
target the search counts, on GPU, how many evaluated values land in each of fifteen nested
windows from 1e-1 down to 1e-15.  The local density is read off an *annulus* -- so the exact-
match spike at an expressible target cannot inflate its own chance model -- and the expected
number of chance matches is the measured density times the window.  The verification precision
``P`` is then **derived** from that measurement as the precision at which the expected chance
count falls below a declared threshold; nothing here hardcodes 60 or 120 digits and hopes.

**A chance model is not enough, and this search proved it.**  The first full run promoted 130
candidates that beat the measured chance model by a mile and were reducible by nothing.  They
were all the same thing: ``2 atan(exp(exp(exp(3/2))))`` reproduces pi to 38 digits, because
``atan`` saturates at ``pi/2``.  No density model can see that -- those values are not random
draws, they are a family converging on the target.  What separates them from real identities
is the digit holdout itself: a true identity's agreement grows with the working precision,
while an approximation stalls at the digit count its own error term allows.  So a survivor
must clear **two** bars at both verification precisions: the chance bar derived from the
measured density, and a holdout bar of agreement within a declared guard of the working
precision.  The receipt names which bar each rejection failed.

**Decoys calibrate the gate.**  Every real target is paired with a seeded decoy: a real number
with no known closed form, sitting in the same neighbourhood, generated from a declared seed.
At the 1e-12 match window the decoys draw *nothing* -- a strong result, but one that leaves
the gate untested, since a gate never handed a chance match has not been shown to reject one.
So the gate is also stress-tested on the annulus [1e-14, 1e-8), where the chance matches
actually live, and the decoys **must** produce zero survivors there.  A surviving decoy means
the gate is too loose and the run aborts.  The decoy column is what makes the real column mean
anything.

Deduplication is measured, not assumed: values are hashed at a declared 40-bit mantissa
quantisation and the distinct-value count is estimated by HyperLogLog with a declared standard
error, cross-checked against an exact distinct count on a declared sub-sample.  The ratio of
distinct values to programs is a real measurement of the grammar's redundancy.

Downstream, every post-gate survivor is run through the existing prior-art screening
(:mod:`sigma_theory_compiler.cf_prior_art_corpus`) and the existing classical reduction
(:mod:`sigma_theory_compiler.cf_proof_router`'s hypergeometric machinery, plus PSLQ against
the classical basis and a symbolic identity proof).  A survivor that reduces to a classical
family is ``KNOWN_BY_PROOF_FAMILY``.  The headline is the count that survives the chance gate
**and** the corpus **and** classical reduction -- and an explicit zero is an honest headline.

**And a null here is not allowed to be silent about itself.**  A headline of zero used to be
one word covering two facts -- the object does not exist, or this grammar cannot spell it --
and no amount of scale separates them.  Every receipt this module seals now carries a
``reachability`` block built by :mod:`sigma_theory_compiler.certified_null_search`: one row per
declared target per mode, each either an explicit witness program in the real ordinal space
proved *exactly* equal to the target, or an honest ``UNRESOLVED``.  A target with a witness
whose sweep found nothing is a ``REAL_NEGATIVE`` and publishable under I6; a target without one
is an ``UNINFORMATIVE_NULL`` and is not a result.  :func:`validate_receipt` refuses a receipt
that reports zero candidates and carries no such block, and the in-campaign
:func:`tamper_control` probes that refusal on every run.

Nothing here is a novelty claim.  ``unreduced_is_not_novel_it_is_unreviewed`` is a sealed claim
of every receipt this module writes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import sympy as sp

from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-compositional-expression-search-1.0"
CHECKPOINT_SCHEMA = "invariant-compositional-expression-search-checkpoint-1.0"


class CompositionalSearchError(ValueError):
    """Raised on malformed input, decode error, failed control, or receipt tamper."""


# ---------------------------------------------------------------------------
# Token alphabet -- declared exactly, closed, and reproducible
# ---------------------------------------------------------------------------

INTEGER_TOKENS: tuple[str, ...] = ("1", "2", "3", "4", "5")
RATIONAL_TOKENS: tuple[str, ...] = ("1/2", "1/3", "2/3", "3/2")
VARIABLE_TOKEN = "k"
UNARY_TOKENS: tuple[str, ...] = ("neg", "recip", "sqr", "sqrt", "exp", "ln", "sin", "cos", "atan")
BINARY_TOKENS: tuple[str, ...] = ("add", "sub", "mul", "div", "pow")

#: The full alphabet in index order.  Indices matter: they are the mixed-radix digits.
TOKEN_NAMES: tuple[str, ...] = (
    *INTEGER_TOKENS,
    *RATIONAL_TOKENS,
    VARIABLE_TOKEN,
    *UNARY_TOKENS,
    *BINARY_TOKENS,
)
TOKEN_COUNT = len(TOKEN_NAMES)  # 24
TERMINAL_COUNT = len(INTEGER_TOKENS) + len(RATIONAL_TOKENS) + 1  # 10, including k
UNARY_START = TERMINAL_COUNT
BINARY_START = TERMINAL_COUNT + len(UNARY_TOKENS)
VARIABLE_INDEX = TERMINAL_COUNT - 1  # index 9 == k

#: Exact rational value of every constant terminal, as a Fraction.  ``k`` has no fixed value.
TERMINAL_FRACTIONS: tuple[Fraction, ...] = tuple(
    Fraction(text) for text in (*INTEGER_TOKENS, *RATIONAL_TOKENS)
)
TERMINAL_FLOATS: tuple[float, ...] = tuple(float(value) for value in TERMINAL_FRACTIONS)

TOKEN_ARITY: tuple[int, ...] = tuple(
    0 if index < TERMINAL_COUNT else (1 if index < BINARY_START else 2)
    for index in range(TOKEN_COUNT)
)

#: Declared stack depth.  A program needing depth 7 is outside the space by construction.
STACK_DEPTH = 6

#: Any intermediate whose magnitude exceeds this, or which is not finite, kills the program.
VALUE_CAP = 1e100

#: ``sin`` and ``cos`` are refused past this argument magnitude.  This is not squeamishness:
#: the spacing of doubles at 2^12 is 9.1e-13, so beyond it a double no longer determines the
#: value of sin to the 1e-12 match tolerance, and two correct implementations disagree by more
#: than the match window purely through argument representation.  Admitting such programs
#: would make the search's own floating-point arithmetic the source of its matches.  The bound
#: is enforced identically in the CUDA kernel, the numpy reference machine, and mpmath.
TRIG_ARGUMENT_BOUND = 4096.0

#: Digit -> token index for each mode.  Mode C simply drops ``k`` from the alphabet, so its
#: space contains no program that could depend on an index variable.
MODE_C_DIGITS: tuple[int, ...] = tuple(i for i in range(TOKEN_COUNT) if i != VARIABLE_INDEX)
MODE_F_DIGITS: tuple[int, ...] = tuple(range(TOKEN_COUNT))

MODE_CONFIG: dict[str, dict[str, Any]] = {
    "C": {
        "description": "constant hunt: programs with no index variable, evaluated to one real",
        "program_length": 9,
        "digit_to_token": list(MODE_C_DIGITS),
        "alphabet_size": len(MODE_C_DIGITS),
        "space_size": len(MODE_C_DIGITS) ** 9,
        "requires_variable": False,
        "submodes": ["constant"],
    },
    "F": {
        "description": (
            "function hunt: programs mentioning k, read as the term of Sum_{k>=1} t_k and as "
            "the sequence a_k = t_k"
        ),
        "program_length": 8,
        "digit_to_token": list(MODE_F_DIGITS),
        "alphabet_size": len(MODE_F_DIGITS),
        "space_size": len(MODE_F_DIGITS) ** 8,
        "requires_variable": True,
        "submodes": ["series", "limit"],
    },
}

SUBMODES: tuple[str, ...] = ("constant", "series", "limit")


# ---------------------------------------------------------------------------
# Search configuration -- changing any value changes the claim and the receipt hash
# ---------------------------------------------------------------------------

#: Nested windows around every target, widest first.  Distance is the scaled absolute
#: residual ``|v - t| / max(1, |t|)``, so one window ladder means the same thing for a target
#: near 0.58 and for one near 23.1.
WINDOW_EXPONENTS: tuple[int, ...] = tuple(range(1, 16))
WINDOWS: tuple[float, ...] = tuple(10.0**-exponent for exponent in WINDOW_EXPONENTS)
#: The declared fp64 match tolerance; every match count and chance figure refers to this.
MATCH_WINDOW_EXPONENT = 12
MATCH_WINDOW = 10.0**-MATCH_WINDOW_EXPONENT
#: Where an *exact* identity in this grammar lands in fp64: a few tens of ulps at worst.
CORE_WINDOW_EXPONENT = 14
CORE_WINDOW = 10.0**-CORE_WINDOW_EXPONENT
#: The chance-match band the gate is stress-tested on.  At the 1e-12 match window the decoys
#: draw literally nothing, which proves the window is clean but leaves the precision gate
#: untested.  Verifying the annulus [1e-14, 1e-8) instead puts real chance matches -- tens per
#: decoy -- through the gate and measures what comes out the other side.
STRESS_WINDOW_EXPONENT = 8
STRESS_WINDOW = 10.0**-STRESS_WINDOW_EXPONENT

#: Mode F series/limit checkpoints.  Partial sums and terms are sampled on a geometric ladder
#: and Richardson-extrapolated in 1/N; the ladder is model-free for algebraic tails and inert
#: for geometric ones.  Stage 1 screens everything cheaply; stage 2 re-runs the near-target set
#: at 64x the depth so the fine windows are exact rather than extrapolated.
STAGE1_TERMS = 64
STAGE1_LADDER = 4
STAGE2_TERMS = 4096
STAGE2_LADDER = 7
#: A Mode F value is admitted only if two Richardson columns agree to this.  Series that fail
#: are excluded and counted: a declared recall limitation, never a correctness risk.  The two
#: stages carry different bars for the same reason they carry different depths -- at 64 terms
#: the Basel series is only self-consistent to about 1e-5, and demanding 1e-13 there would
#: throw away the very series the search exists to find.  The strict bar is applied at stage 2,
#: where it is also what makes the fine window counts mean anything.
STAGE1_STABILITY = 1e-4
STAGE2_STABILITY = 1e-13
EXTRAPOLATION_STABILITY = STAGE2_STABILITY
#: Mode F window counts strictly inside this exponent are re-derived from the stage-2 values,
#: so the fine end of the ladder is exact rather than a stage-1 extrapolation artefact.
STAGE1_WINDOW_FLOOR_EXPONENT = 3

#: HyperLogLog register count for the distinct-value estimate.  Standard error 1.04/sqrt(m).
HLL_PRECISION = 14
HLL_REGISTERS = 1 << HLL_PRECISION
#: Values are quantised to this many mantissa bits before hashing (~12 significant decimals).
VALUE_HASH_MANTISSA_BITS = 40

#: Chance-match accounting.  A survivor must beat chance by this margin: the expected number
#: of chance matches at the verification tolerance must fall below the threshold.
CHANCE_THRESHOLD = 1e-6
#: Density is read from the smallest annulus holding at least this many values, so the exact-
#: match spike at an expressible target cannot inflate its own chance model.
DENSITY_MIN_COUNT = 30
#: Annuli at or inside this window are never used for density (they carry the signal spike).
DENSITY_MAX_EXPONENT = 2
#: Guard digits added to the derived agreement requirement to set the mpmath working precision.
VERIFY_GUARD_DIGITS = 30
#: The chance model is not the only way a candidate can be an accident.  This search found a
#: whole family of *asymptotic* near-identities -- ``2 atan(exp(exp(exp(3/2))))`` reproduces pi
#: to 38 digits because ``atan`` saturates at ``pi/2``, and no amount of chance accounting sees
#: it, because those values are not random draws.  The rule that separates them from real
#: identities is the digit holdout itself: a true identity agrees to within a few guard digits
#: of the *working precision*, so its agreement grows when the precision grows, while an
#: approximation stalls at a fixed digit count.  A survivor must therefore agree to
#: ``working_dps - AGREEMENT_GUARD_DIGITS`` at both verification precisions, on top of beating
#: the measured chance model.
AGREEMENT_GUARD_DIGITS = 15
#: Floor on the derived precision.  The derivation may demand more; it may never demand less.
VERIFY_DPS_FLOOR = 60

#: Seed for every decoy target and for the deterministic samples.  Declared, never drawn.
DECOY_SEED = "invariant-compositional-expression-search-decoys-v1"
#: Each real target gets one paired decoy this far away (same neighbourhood, no closed form).
DECOY_PAIR_OFFSET_RANGE = (1e-3, 1e-2)
#: Plus this many decoys drawn uniformly across the span of the real targets.
DECOY_UNIFORM_COUNT = 4

#: GPU launch geometry.  One launch must stay well under the Windows TDR watchdog, so the
#: checkpoint chunk is many launches and the launch itself is small.
DEFAULT_LAUNCH_SIZE = 1 << 26
DEFAULT_CHUNK_SIZE = 1 << 32
DEFAULT_THREADS_PER_BLOCK = 256
#: Per-launch retention buffer.  On overflow the launch is split and re-run, so retention is
#: deterministic and complete rather than racy.
HIT_CAPACITY = 1 << 21

#: Retention runs as a second full sweep whose per-slot windows and sampling rates are derived
#: from the first sweep's exact counts.  Two tiers: the *core* tier covers the window where an
#: exact identity must land, the *band* tier samples the rest of the match window so the claim
#: "the band carries no verified signal" is measured rather than assumed.  A tier whose derived
#: rate is 1.0 is complete for that slot, and the receipt says so per slot.
RETENTION_BUDGET_CORE = 1024
RETENTION_BUDGET_BAND = 512
#: Mode F stage-1 values are only good to the stage-1 stability bar, so its core tier is the
#: near-target set that stage 2 then re-evaluates at 64x the depth.  The near window is ten
#: times that bar on purpose: a series whose true value matches a target cannot fall out of
#: the near-target set through stage-1 extrapolation error alone.
MODE_F_NEAR_WINDOW = 1e-3
RETENTION_BUDGET_MODE_F_NEAR = 1 << 24
#: Deterministic CPU/GPU cross-check sample size.
CROSSCHECK_SAMPLE = 65536
#: Exact-distinct cross-check of the HyperLogLog estimate runs over this many ordinals.
DEDUP_EXACT_SAMPLE = 1 << 21

CLAIMS = {
    "grammar_is_generative_not_curated": True,
    "chance_match_gate_calibrated_by_decoys": True,
    "unreduced_is_not_novel_it_is_unreviewed": True,
    "corpus_absence_establishes_novelty": False,
    "enumerated_count_is_measured_not_extrapolated": True,
    "match_at_fit_precision_is_not_discovery": True,
    "proof_by_classical_family_implies_known": True,
    "survivor_is_conjecture_not_theorem": True,
}


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

#: Real targets.  ``sympy`` gives the classical reduction lane an exact symbolic form and
#: ``mpmath`` gives the verification lane an arbitrary-precision value.
REAL_TARGETS: tuple[dict[str, str], ...] = (
    {"name": "pi", "definition": "pi"},
    {"name": "e", "definition": "exp(1)"},
    {"name": "sqrt2", "definition": "sqrt(2)"},
    {"name": "sqrt3", "definition": "sqrt(3)"},
    {"name": "ln2", "definition": "log(2)"},
    {"name": "ln3", "definition": "log(3)"},
    {"name": "zeta2", "definition": "pi^2/6"},
    {"name": "zeta3", "definition": "zeta(3)"},
    {"name": "catalan", "definition": "Catalan"},
    {"name": "euler_gamma", "definition": "EulerGamma"},
    {"name": "phi", "definition": "(1 + sqrt(5))/2"},
    {"name": "e_pi", "definition": "exp(pi)"},
)

SYMBOLIC_TARGETS: dict[str, sp.Expr] = {
    "pi": sp.pi,
    "e": sp.E,
    "sqrt2": sp.sqrt(2),
    "sqrt3": sp.sqrt(3),
    "ln2": sp.log(2),
    "ln3": sp.log(3),
    "zeta2": sp.pi**2 / 6,
    "zeta3": sp.zeta(3),
    "catalan": sp.Catalan,
    "euler_gamma": sp.EulerGamma,
    "phi": (1 + sp.sqrt(5)) / 2,
    "e_pi": sp.exp(sp.pi),
}


def target_value_mp(name: str) -> mp.mpf:
    """Exact real target at the current mpmath working precision."""

    if name == "pi":
        return +mp.pi
    if name == "e":
        return +mp.e
    if name == "sqrt2":
        return mp.sqrt(2)
    if name == "sqrt3":
        return mp.sqrt(3)
    if name == "ln2":
        return mp.log(2)
    if name == "ln3":
        return mp.log(3)
    if name == "zeta2":
        return mp.pi**2 / 6
    if name == "zeta3":
        return mp.zeta(3)
    if name == "catalan":
        return +mp.catalan
    if name == "euler_gamma":
        return +mp.euler
    if name == "phi":
        return +mp.phi
    if name == "e_pi":
        return mp.exp(mp.pi)
    raise CompositionalSearchError(f"unknown real target: {name}")


def _seeded_unit(seed: str, label: str) -> float:
    """Deterministic float in [0, 1) from SHA-256 of the declared seed and a label."""

    digest = hashlib.sha256(f"{seed}|{label}".encode()).digest()
    return int.from_bytes(digest[:7], "big") / float(1 << 56)


def build_decoy_targets() -> tuple[dict[str, Any], ...]:
    """Decoys: reals with no known closed form, generated from the declared seed.

    Two kinds.  *Paired* decoys sit a seeded 1e-3..1e-2 away from each real target, so they
    sample the same local value density that the real target sits in -- the only way a chance
    count is comparable.  *Uniform* decoys are drawn across the span of the real targets and
    catch the case where the paired offsets accidentally land somewhere unusual.
    """

    with mp.workdps(60):
        real_values = [float(target_value_mp(item["name"])) for item in REAL_TARGETS]
    low, high = min(real_values), max(real_values)
    decoys: list[dict[str, Any]] = []
    for item, value in zip(REAL_TARGETS, real_values, strict=True):
        unit = _seeded_unit(DECOY_SEED, f"pair|{item['name']}")
        span_low, span_high = DECOY_PAIR_OFFSET_RANGE
        offset = span_low + unit * (span_high - span_low)
        sign = 1.0 if _seeded_unit(DECOY_SEED, f"sign|{item['name']}") < 0.5 else -1.0
        decoys.append(
            {
                "name": f"decoy_near_{item['name']}",
                "kind": "paired",
                "paired_with": item["name"],
                "value": value + sign * offset,
                "definition": (
                    f"{item['definition']} {'+' if sign > 0 else '-'} "
                    f"{format(offset, '.17e')} (seeded, no known closed form)"
                ),
            }
        )
    for index in range(DECOY_UNIFORM_COUNT):
        unit = _seeded_unit(DECOY_SEED, f"uniform|{index}")
        decoys.append(
            {
                "name": f"decoy_uniform_{index}",
                "kind": "uniform",
                "paired_with": None,
                "value": low + unit * (high - low),
                "definition": (
                    f"{format(low, '.17e')} + u*{format(high - low, '.17e')}, u from "
                    "SHA-256 of the declared seed (no known closed form)"
                ),
            }
        )
    return tuple(decoys)


def build_target_slots() -> tuple[dict[str, Any], ...]:
    """The full slot table the GPU screens against: real targets first, then decoys."""

    slots: list[dict[str, Any]] = []
    with mp.workdps(60):
        for item in REAL_TARGETS:
            slots.append(
                {
                    "slot": len(slots),
                    "name": item["name"],
                    "role": "real",
                    "definition": item["definition"],
                    "value": float(target_value_mp(item["name"])),
                    "paired_with": None,
                }
            )
    for decoy in build_decoy_targets():
        slots.append(
            {
                "slot": len(slots),
                "name": decoy["name"],
                "role": "decoy",
                "definition": decoy["definition"],
                "value": float(decoy["value"]),
                "paired_with": decoy["paired_with"],
            }
        )
    return tuple(slots)


TARGET_SLOTS = build_target_slots()
SLOT_COUNT = len(TARGET_SLOTS)
SLOT_VALUES = np.array([slot["value"] for slot in TARGET_SLOTS], dtype=np.float64)


def decoy_value_mp(name: str) -> mp.mpf:
    """A decoy's value at working precision: exactly the fp64 number that was screened.

    A decoy has no closed form by construction, so its arbitrary-precision value *is* its
    double.  Verifying a candidate against it at 200 digits therefore asks the only honest
    question available: does the expression reproduce those 17 digits and then keep going?
    """

    for slot in TARGET_SLOTS:
        if slot["name"] == name and slot["role"] == "decoy":
            return mp.mpf(slot["value"])
    raise CompositionalSearchError(f"unknown decoy target: {name}")


def slot_value_mp(slot: Mapping[str, Any]) -> mp.mpf:
    """Working-precision value of any slot, real or decoy."""

    if slot["role"] == "real":
        return target_value_mp(str(slot["name"]))
    return decoy_value_mp(str(slot["name"]))


# ---------------------------------------------------------------------------
# Ordinal codec
# ---------------------------------------------------------------------------


def mode_digits(mode: str) -> tuple[int, ...]:
    if mode not in MODE_CONFIG:
        raise CompositionalSearchError(f"unknown mode: {mode}")
    return tuple(MODE_CONFIG[mode]["digit_to_token"])


def decode_ordinal(ordinal: int, mode: str) -> tuple[int, ...]:
    """Ordinal -> token sequence, mixed-radix little-endian over the mode's alphabet.

    Digit 0 is the first token executed, so the ordinal's least significant digit is the
    program's first instruction.  The map is a bijection onto ``base^L`` token sequences; the
    validity pass in :func:`program_status` is what makes it an injection onto programs.
    """

    config = MODE_CONFIG.get(mode)
    if config is None:
        raise CompositionalSearchError(f"unknown mode: {mode}")
    if not 0 <= ordinal < config["space_size"]:
        raise CompositionalSearchError(f"ordinal out of range for mode {mode}: {ordinal}")
    digits = mode_digits(mode)
    base = len(digits)
    tokens: list[int] = []
    value = ordinal
    for _ in range(config["program_length"]):
        tokens.append(digits[value % base])
        value //= base
    return tuple(tokens)


def encode_program(tokens: Sequence[int], mode: str) -> int:
    """Inverse of :func:`decode_ordinal`; used by the controls and the tests."""

    config = MODE_CONFIG.get(mode)
    if config is None:
        raise CompositionalSearchError(f"unknown mode: {mode}")
    if len(tokens) != config["program_length"]:
        raise CompositionalSearchError(
            f"mode {mode} programs have exactly {config['program_length']} tokens"
        )
    digits = mode_digits(mode)
    index_of = {token: index for index, token in enumerate(digits)}
    base = len(digits)
    ordinal = 0
    for token in reversed(list(tokens)):
        if token not in index_of:
            raise CompositionalSearchError(f"token {token} is outside the mode {mode} alphabet")
        ordinal = ordinal * base + index_of[token]
    return ordinal


def program_status(tokens: Sequence[int]) -> str:
    """``ok``, ``underflow``, ``overflow``, or ``residue`` for one token sequence."""

    depth = 0
    for token in tokens:
        arity = TOKEN_ARITY[token]
        if arity == 0:
            if depth >= STACK_DEPTH:
                return "overflow"
            depth += 1
        else:
            if depth < arity:
                return "underflow"
            depth -= arity - 1
    return "ok" if depth == 1 else "residue"


def is_valid_program(tokens: Sequence[int]) -> bool:
    return program_status(tokens) == "ok"


def uses_variable(tokens: Sequence[int]) -> bool:
    return VARIABLE_INDEX in tuple(tokens)


def count_valid_programs(mode: str) -> dict[str, int]:
    """Exact structural counts for a mode by dynamic programming over the stack depth.

    This is a closed-form check on the measured valid fraction: the GPU count and this count
    must agree exactly, which is the strongest available test that the kernel's validity pass
    implements the declared codec.
    """

    config = MODE_CONFIG[mode]
    length = int(config["program_length"])
    digits = mode_digits(mode)
    terminals = sum(1 for token in digits if TOKEN_ARITY[token] == 0)
    unary = sum(1 for token in digits if TOKEN_ARITY[token] == 1)
    binary = sum(1 for token in digits if TOKEN_ARITY[token] == 2)
    variable_terminals = sum(
        1 for token in digits if TOKEN_ARITY[token] == 0 and token == VARIABLE_INDEX
    )
    # state: (depth, saw_variable)
    state: dict[tuple[int, bool], int] = {(0, False): 1}
    for _ in range(length):
        nxt: dict[tuple[int, bool], int] = {}

        def add(key: tuple[int, bool], count: int, table: dict[tuple[int, bool], int]) -> None:
            table[key] = table.get(key, 0) + count

        for (depth, saw), count in state.items():
            if depth + 1 <= STACK_DEPTH:
                if terminals - variable_terminals:
                    add((depth + 1, saw), count * (terminals - variable_terminals), nxt)
                if variable_terminals:
                    add((depth + 1, True), count * variable_terminals, nxt)
            if depth >= 1 and unary:
                add((depth, saw), count * unary, nxt)
            if depth >= 2 and binary:
                add((depth - 1, saw), count * binary, nxt)
        state = nxt
    valid = sum(count for (depth, _), count in state.items() if depth == 1)
    with_variable = sum(
        count for (depth, saw), count in state.items() if depth == 1 and saw
    )
    return {
        "space_size": int(config["space_size"]),
        "structurally_valid": valid,
        "valid_with_variable": with_variable,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_rpn(tokens: Sequence[int]) -> str:
    return " ".join(TOKEN_NAMES[token] for token in tokens)


_INFIX_BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^"}


def render_infix(tokens: Sequence[int]) -> str:
    """Human-readable rendering.  Binary operations are always parenthesised."""

    stack: list[str] = []
    for token in tokens:
        name = TOKEN_NAMES[token]
        arity = TOKEN_ARITY[token]
        if arity == 0:
            stack.append(name)
        elif arity == 1:
            if not stack:
                return "INVALID_PROGRAM"
            operand = stack.pop()
            if name == "neg":
                stack.append(f"(-{operand})")
            elif name == "recip":
                stack.append(f"(1/{operand})")
            elif name == "sqr":
                stack.append(f"({operand})^2")
            else:
                stack.append(f"{name}({operand})")
        else:
            if len(stack) < 2:
                return "INVALID_PROGRAM"
            right = stack.pop()
            left = stack.pop()
            stack.append(f"({left} {_INFIX_BINARY[name]} {right})")
    if len(stack) != 1:
        return "INVALID_PROGRAM"
    return stack[0]


# ---------------------------------------------------------------------------
# Exact evaluation: mpmath (verification) and sympy (classical reduction)
# ---------------------------------------------------------------------------


def evaluate_mp(tokens: Sequence[int], variable: Any = None) -> mp.mpf | None:
    """Arbitrary-precision evaluation at the current working precision, or ``None``.

    ``None`` means the program is structurally invalid, left a domain (``ln`` of a
    non-positive number, ``sqrt`` of a negative, division by zero, a fractional power of a
    negative base), or produced a non-finite value.  The domain rules are exactly the ones the
    GPU kernel enforces, so a survivor cannot be an artefact of the two disagreeing.
    """

    stack: list[mp.mpf] = []
    for token in tokens:
        name = TOKEN_NAMES[token]
        arity = TOKEN_ARITY[token]
        try:
            if arity == 0:
                if token == VARIABLE_INDEX:
                    if variable is None:
                        return None
                    stack.append(mp.mpf(variable))
                else:
                    fraction = TERMINAL_FRACTIONS[token]
                    stack.append(mp.mpf(fraction.numerator) / fraction.denominator)
                continue
            if arity == 1:
                if not stack:
                    return None
                operand = stack.pop()
                if name == "neg":
                    value = -operand
                elif name == "recip":
                    if operand == 0:
                        return None
                    value = 1 / operand
                elif name == "sqr":
                    value = operand * operand
                elif name == "sqrt":
                    if operand < 0:
                        return None
                    value = mp.sqrt(operand)
                elif name == "exp":
                    value = mp.exp(operand)
                elif name == "ln":
                    if operand <= 0:
                        return None
                    value = mp.log(operand)
                elif name == "sin":
                    if abs(operand) > TRIG_ARGUMENT_BOUND:
                        return None
                    value = mp.sin(operand)
                elif name == "cos":
                    if abs(operand) > TRIG_ARGUMENT_BOUND:
                        return None
                    value = mp.cos(operand)
                else:
                    value = mp.atan(operand)
            else:
                if len(stack) < 2:
                    return None
                right = stack.pop()
                left = stack.pop()
                if name == "add":
                    value = left + right
                elif name == "sub":
                    value = left - right
                elif name == "mul":
                    value = left * right
                elif name == "div":
                    if right == 0:
                        return None
                    value = left / right
                else:
                    if left < 0 and right != int(right):
                        return None
                    if left == 0 and right <= 0:
                        return None
                    value = mp.power(left, right)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        if not mp.isfinite(value) or abs(value) > mp.mpf(VALUE_CAP):
            return None
        stack.append(value)
    if len(stack) != 1:
        return None
    return stack[0]


_SYMPY_K = sp.Symbol("k", positive=True)


def to_sympy(tokens: Sequence[int]) -> sp.Expr | None:
    """Exact symbolic form of a program, or ``None`` when it is structurally invalid."""

    stack: list[sp.Expr] = []
    for token in tokens:
        name = TOKEN_NAMES[token]
        arity = TOKEN_ARITY[token]
        if arity == 0:
            if token == VARIABLE_INDEX:
                stack.append(_SYMPY_K)
            else:
                fraction = TERMINAL_FRACTIONS[token]
                stack.append(sp.Rational(fraction.numerator, fraction.denominator))
            continue
        if arity == 1:
            if not stack:
                return None
            operand = stack.pop()
            table = {
                "neg": lambda x: -x,
                "recip": lambda x: 1 / x,
                "sqr": lambda x: x**2,
                "sqrt": sp.sqrt,
                "exp": sp.exp,
                "ln": sp.log,
                "sin": sp.sin,
                "cos": sp.cos,
                "atan": sp.atan,
            }
            stack.append(table[name](operand))
            continue
        if len(stack) < 2:
            return None
        right = stack.pop()
        left = stack.pop()
        table2 = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b,
            "pow": lambda a, b: a**b,
        }
        stack.append(table2[name](left, right))
    if len(stack) != 1:
        return None
    return stack[0]


# ---------------------------------------------------------------------------
# Series and limit semantics at arbitrary precision (the verification lane)
# ---------------------------------------------------------------------------

SERIES_START_INDEX = 1


def _term_function(tokens: Sequence[int]):
    def term(index: Any) -> mp.mpf:
        value = evaluate_mp(tokens, index)
        if value is None:
            raise CompositionalSearchError("term left the declared domain")
        return value

    return term


def evaluate_series_mp(tokens: Sequence[int]) -> mp.mpf | None:
    """``Sum_{k>=1} t_k`` at the current working precision, or ``None`` if it does not sum.

    ``mpmath.nsum`` applies Richardson and Euler-Maclaurin acceleration; the guard in front of
    it rejects terms that do not decay, so a divergent program cannot be handed to an
    accelerator that would happily return a finite analytic continuation of it.
    """

    term = _term_function(tokens)
    try:
        head = [abs(term(index)) for index in (1, 8, 64, 512)]
    except CompositionalSearchError:
        return None
    if not all(mp.isfinite(value) for value in head):
        return None
    if head[3] > head[2] or head[2] > head[1]:
        return None
    if head[3] * 512 > mp.mpf("1e-2") * max(mp.mpf(1), head[1]):
        # decays too slowly for the declared acceleration to be trusted
        return None
    try:
        total = mp.nsum(term, [SERIES_START_INDEX, mp.inf])
    except (CompositionalSearchError, ValueError, ZeroDivisionError, ArithmeticError):
        return None
    if total is None or not mp.isfinite(total):
        return None
    return total


def evaluate_limit_mp(tokens: Sequence[int]) -> mp.mpf | None:
    """``lim_{k->inf} t_k`` at the current working precision, or ``None``."""

    term = _term_function(tokens)
    try:
        value = mp.limit(term, mp.inf)
    except (CompositionalSearchError, ValueError, ZeroDivisionError, ArithmeticError):
        return None
    if value is None or not mp.isfinite(value):
        return None
    try:
        far = term(mp.mpf(10) ** 12)
    except CompositionalSearchError:
        return None
    if abs(far - value) > mp.mpf("1e-6") * max(mp.mpf(1), abs(value)):
        return None
    return value


def evaluate_program_value_mp(tokens: Sequence[int], submode: str) -> mp.mpf | None:
    """One dispatch point for all three submodes, used by verification and by the controls."""

    if submode == "constant":
        return evaluate_mp(tokens)
    if submode == "series":
        return evaluate_series_mp(tokens)
    if submode == "limit":
        return evaluate_limit_mp(tokens)
    raise CompositionalSearchError(f"unknown submode: {submode}")


# ---------------------------------------------------------------------------
# CUDA stack machine
# ---------------------------------------------------------------------------

NSTATUS = 8
NMAG = 96
STATUS_LABELS = (
    "processed",
    "structurally_valid",
    "domain_or_overflow_rejected",
    "usable_values",
    "structurally_invalid",
    "valid_with_variable",
    "series_admitted",
    "limit_admitted",
)


def _checkpoints(terms: int, ladder: int) -> tuple[int, ...]:
    """Geometric ladder of sample points ending at ``terms``, e.g. 8, 16, 32, 64."""

    return tuple(terms >> (ladder - 1 - index) for index in range(ladder))


def _cuda_prelude(mode: str, submode_count: int) -> str:
    digits = mode_digits(mode)
    length = int(MODE_CONFIG[mode]["program_length"])
    digit_list = ", ".join(str(token) for token in digits)
    term_list = ", ".join(format(value, ".17g") for value in TERMINAL_FLOATS)
    window_list = ", ".join(format(value, ".17g") for value in WINDOWS)
    return f"""
#define PROGRAM_LENGTH {length}
#define BASE {len(digits)}
#define NTERM {TERMINAL_COUNT}
#define BINSTART {BINARY_START}
#define VARIDX {VARIABLE_INDEX}
#define STACK_DEPTH {STACK_DEPTH}
#define VALUE_CAP {VALUE_CAP:.17g}
#define TRIG_BOUND {TRIG_ARGUMENT_BOUND:.17g}
#define NSLOT {SLOT_COUNT}
#define NWIN {len(WINDOWS)}
#define NSTATUS {NSTATUS}
#define NMAG {NMAG}
#define NSUB {submode_count}
#define HLL_P {HLL_PRECISION}
#define HLL_M {HLL_REGISTERS}
#define MANTISSA_SCALE {float(1 << VALUE_HASH_MANTISSA_BITS):.17g}
#define STABILITY1 {STAGE1_STABILITY:.17g}
#define STABILITY2 {STAGE2_STABILITY:.17g}

__device__ const int DIGIT2TOK[BASE] = {{{digit_list}}};
__device__ const double TERMVAL[NTERM - 1] = {{{term_list}}};
__device__ const double WINDOW[NWIN] = {{{window_list}}};

__device__ __forceinline__ unsigned long long pack_program(long long ordinal) {{
    unsigned long long packed = 0ULL;
    long long rest = ordinal;
    #pragma unroll
    for (int i = 0; i < PROGRAM_LENGTH; ++i) {{
        int d = (int)(rest % BASE);
        rest /= BASE;
        packed |= ((unsigned long long)DIGIT2TOK[d]) << (5 * i);
    }}
    return packed;
}}

__device__ __forceinline__ int token_at(unsigned long long packed, int index) {{
    return (int)((packed >> (5 * index)) & 31ULL);
}}

__device__ __forceinline__ int structural_ok(unsigned long long packed) {{
    int depth = 0;
    #pragma unroll
    for (int i = 0; i < PROGRAM_LENGTH; ++i) {{
        int t = token_at(packed, i);
        int arity = (t < NTERM) ? 0 : ((t < BINSTART) ? 1 : 2);
        if (arity == 0) {{
            if (depth >= STACK_DEPTH) return 0;
            depth += 1;
        }} else {{
            if (depth < arity) return 0;
            depth -= arity - 1;
        }}
    }}
    return depth == 1 ? 1 : 0;
}}

__device__ __forceinline__ int has_variable(unsigned long long packed) {{
    #pragma unroll
    for (int i = 0; i < PROGRAM_LENGTH; ++i) {{
        if (token_at(packed, i) == VARIDX) return 1;
    }}
    return 0;
}}

__device__ __forceinline__ double apply_unary(int op, double a, int* bad) {{
    switch (op) {{
        case 0: return -a;
        case 1: if (a == 0.0) {{ *bad = 1; return 0.0; }} return 1.0 / a;
        case 2: return a * a;
        case 3: if (a < 0.0) {{ *bad = 1; return 0.0; }} return sqrt(a);
        case 4: return exp(a);
        case 5: if (a <= 0.0) {{ *bad = 1; return 0.0; }} return log(a);
        case 6: if (fabs(a) > TRIG_BOUND) {{ *bad = 1; return 0.0; }} return sin(a);
        case 7: if (fabs(a) > TRIG_BOUND) {{ *bad = 1; return 0.0; }} return cos(a);
        default: return atan(a);
    }}
}}

__device__ __forceinline__ double apply_binary(int op, double a, double b, int* bad) {{
    switch (op) {{
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: if (b == 0.0) {{ *bad = 1; return 0.0; }} return a / b;
        default:
            if (a < 0.0 && b != trunc(b)) {{ *bad = 1; return 0.0; }}
            if (a == 0.0 && b <= 0.0) {{ *bad = 1; return 0.0; }}
            return pow(a, b);
    }}
}}

/* Runs a structurally valid program.  Returns 0 on success and 1 when the program left the
   declared domain or exceeded the magnitude cap. */
__device__ __forceinline__ int run_program(unsigned long long packed, double kv, double* out) {{
    double st[STACK_DEPTH];
    int sp = 0;
    for (int i = 0; i < PROGRAM_LENGTH; ++i) {{
        int t = token_at(packed, i);
        double v;
        if (t < NTERM) {{
            st[sp++] = (t == VARIDX) ? kv : TERMVAL[t];
            continue;
        }}
        int bad = 0;
        if (t < BINSTART) {{
            v = apply_unary(t - NTERM, st[sp - 1], &bad);
            if (bad) return 1;
            st[sp - 1] = v;
        }} else {{
            v = apply_binary(t - BINSTART, st[sp - 2], st[sp - 1], &bad);
            if (bad) return 1;
            sp -= 1;
            st[sp - 1] = v;
        }}
        if (!isfinite(v) || fabs(v) > VALUE_CAP) return 1;
    }}
    *out = st[0];
    return 0;
}}

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {{
    x += 0x9E3779B97F4A7C15ULL;
    unsigned long long z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}}

/* Quantise to the declared mantissa precision, then hash.  Two values that agree to ~12
   significant decimals collide on purpose: that is the declared dedup precision. */
__device__ __forceinline__ unsigned long long value_key(double v) {{
    int e;
    double m = frexp(v, &e);
    long long mi = (long long)llrint(m * MANTISSA_SCALE);
    unsigned long long ue = (unsigned long long)((unsigned int)(e + 4096)) & 0x1FFFULL;
    unsigned long long um = (unsigned long long)(mi + (1LL << 41)) & 0x3FFFFFFFFFFULL;
    return (ue << 44) | um;
}}

__device__ __forceinline__ void hll_update(unsigned int* hll, int bank, double v) {{
    unsigned long long h = splitmix64(value_key(v));
    unsigned int idx = (unsigned int)(h >> (64 - HLL_P));
    unsigned long long w = (h << HLL_P) | (1ULL << (HLL_P - 1));
    unsigned int rank = (unsigned int)(__clzll(w) + 1);
    unsigned int* slot = &hll[bank * HLL_M + idx];
    if (*slot < rank) atomicMax(slot, rank);
}}

__device__ __forceinline__ int magnitude_bucket(double v) {{
    if (v == 0.0) return 0;
    int e = ilogb(fabs(v));
    int b = (e + 352) / 8;
    if (b < 0) b = 0;
    if (b >= NMAG) b = NMAG - 1;
    return b;
}}

/* Richardson extrapolation in 1/N over a geometric ladder.  Model-free for algebraic tails,
   inert for geometric ones, and the residual between the last two columns is the stability
   estimate that decides admission. */
__device__ __forceinline__ double richardson(double* T, int npts, double* stability) {{
    double prev = T[npts - 1];
    for (int m = 1; m <= npts - 1; ++m) {{
        double f = (double)(1 << m);
        for (int j = 0; j + m <= npts - 1; ++j) {{
            T[j] = (f * T[j + 1] - T[j]) / (f - 1.0);
        }}
        if (m == npts - 2) prev = T[1];
    }}
    *stability = fabs(T[0] - prev);
    return T[0];
}}
"""


_ACCUMULATE_BODY = """
__device__ __forceinline__ void emit(
    int tag,
    long long ordinal,
    double v,
    long long* hit_ordinal,
    int* hit_slot,
    double* hit_value,
    int* hit_count,
    int hit_capacity)
{
    int at = atomicAdd(hit_count, 1);
    if (at < hit_capacity) {
        hit_ordinal[at] = ordinal;
        hit_slot[at] = tag;
        hit_value[at] = v;
    }
}

/* One evaluated value against every target slot.  ``retain_*`` are per-(submode, slot) arrays
   derived from the census sweep: a value is kept when it is inside that slot's tier window and
   its ordinal hash falls under that tier's sampling threshold.  Both are declared, so the
   retained set is a deterministic function of the ordinal, not of thread scheduling. */
__device__ __forceinline__ void account(
    int sub,
    double v,
    long long ordinal,
    const double* sh_targets,
    const double* sh_scale,
    unsigned long long* sh_win,
    unsigned long long* sh_mag,
    unsigned int* hll,
    const double* retain_window_core,
    const unsigned long long* retain_threshold_core,
    const double* retain_window_band,
    const unsigned long long* retain_threshold_band,
    long long* hit_ordinal,
    int* hit_slot,
    double* hit_value,
    int* hit_count,
    int hit_capacity)
{
    hll_update(hll, sub, v);
    atomicAdd(&sh_mag[sub * NMAG + magnitude_bucket(v)], 1ULL);
    for (int s = 0; s < NSLOT; ++s) {
        double d = fabs(v - sh_targets[s]) * sh_scale[s];
        if (d >= WINDOW[0]) continue;
        int index = sub * NSLOT + s;
        for (int w = 0; w < NWIN; ++w) {
            if (d < WINDOW[w]) atomicAdd(&sh_win[index * NWIN + w], 1ULL);
        }
        unsigned long long h = splitmix64((unsigned long long)ordinal * 0x2545F4914F6CDD1DULL
                                          + (unsigned long long)(index + 1));
        if (d < retain_window_core[index] && h < retain_threshold_core[index]) {
            emit(index * 2, ordinal, v, hit_ordinal, hit_slot, hit_value, hit_count,
                 hit_capacity);
        }
        if (d < retain_window_band[index] && splitmix64(h) < retain_threshold_band[index]) {
            emit(index * 2 + 1, ordinal, v, hit_ordinal, hit_slot, hit_value, hit_count,
                 hit_capacity);
        }
    }
}
"""


_KERNEL_ARGS = """
    const long long start,
    const int count,
    const double* __restrict__ targets,
    const double* __restrict__ scales,
    unsigned long long* __restrict__ window_counts,
    unsigned long long* __restrict__ status_counts,
    unsigned long long* __restrict__ mag_hist,
    unsigned int* __restrict__ hll,
    const double* __restrict__ retain_window_core,
    const unsigned long long* __restrict__ retain_threshold_core,
    const double* __restrict__ retain_window_band,
    const unsigned long long* __restrict__ retain_threshold_band,
    long long* __restrict__ hit_ordinal,
    int* __restrict__ hit_slot,
    double* __restrict__ hit_value,
    int* __restrict__ hit_count,
    const int hit_capacity)
"""

_KERNEL_SHARED_SETUP = """
    __shared__ unsigned long long sh_win[NSUB * NSLOT * NWIN];
    __shared__ unsigned long long sh_status[NSTATUS];
    __shared__ unsigned long long sh_mag[NSUB * NMAG];
    __shared__ double sh_targets[NSLOT];
    __shared__ double sh_scale[NSLOT];
    for (int i = threadIdx.x; i < NSUB * NSLOT * NWIN; i += blockDim.x) sh_win[i] = 0ULL;
    for (int i = threadIdx.x; i < NSTATUS; i += blockDim.x) sh_status[i] = 0ULL;
    for (int i = threadIdx.x; i < NSUB * NMAG; i += blockDim.x) sh_mag[i] = 0ULL;
    for (int i = threadIdx.x; i < NSLOT; i += blockDim.x) {
        sh_targets[i] = targets[i];
        sh_scale[i] = scales[i];
    }
    __syncthreads();
"""

_KERNEL_FLUSH = """
    __syncthreads();"""

_ACCOUNT_CALL = """account(SUBMODE, VALUE, ordinal, sh_targets, sh_scale, sh_win, sh_mag, hll,
                        retain_window_core, retain_threshold_core, retain_window_band,
                        retain_threshold_band, hit_ordinal, hit_slot, hit_value, hit_count,
                        hit_capacity);"""


def _account_call(submode: int, value: str) -> str:
    return _ACCOUNT_CALL.replace("SUBMODE", str(submode)).replace("VALUE", value)


_MODE_C_KERNEL = (
    """
extern "C" __global__ void eval_mode_c("""
    + _KERNEL_ARGS
    + "{"
    + _KERNEL_SHARED_SETUP
    + """
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        long long ordinal = start + tid;
        unsigned long long packed = pack_program(ordinal);
        atomicAdd(&sh_status[0], 1ULL);
        if (!structural_ok(packed)) {
            atomicAdd(&sh_status[4], 1ULL);
        } else {
            atomicAdd(&sh_status[1], 1ULL);
            double v;
            if (run_program(packed, 0.0, &v) != 0) {
                atomicAdd(&sh_status[2], 1ULL);
            } else {
                atomicAdd(&sh_status[3], 1ULL);
                """
    + _account_call(0, "v")
    + """
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NSUB * NSLOT * NWIN; i += blockDim.x) {
        if (sh_win[i]) atomicAdd(&window_counts[i], sh_win[i]);
    }
    for (int i = threadIdx.x; i < NSTATUS; i += blockDim.x) {
        if (sh_status[i]) atomicAdd(&status_counts[i], sh_status[i]);
    }
    for (int i = threadIdx.x; i < NSUB * NMAG; i += blockDim.x) {
        if (sh_mag[i]) atomicAdd(&mag_hist[i], sh_mag[i]);
    }
}

extern "C" __global__ void eval_list_mode_c(
    const long long* __restrict__ ordinals,
    const int count,
    double* __restrict__ values,
    unsigned char* __restrict__ flags)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    unsigned long long packed = pack_program(ordinals[tid]);
    values[tid] = 0.0;
    flags[tid] = 0;
    if (!structural_ok(packed)) return;
    double v;
    if (run_program(packed, 0.0, &v) != 0) return;
    values[tid] = v;
    flags[tid] = 1;
}
"""
)


def _cuda_mode_f_kernels() -> str:
    s1_check = ", ".join(str(value) for value in _checkpoints(STAGE1_TERMS, STAGE1_LADDER))
    s2_check = ", ".join(str(value) for value in _checkpoints(STAGE2_TERMS, STAGE2_LADDER))
    body = f"""
#define S1_TERMS {STAGE1_TERMS}
#define S1_PTS {STAGE1_LADDER}
#define S2_TERMS {STAGE2_TERMS}
#define S2_PTS {STAGE2_LADDER}
__device__ const int S1_CHECK[S1_PTS] = {{{s1_check}}};
__device__ const int S2_CHECK[S2_PTS] = {{{s2_check}}};

extern "C" __global__ void eval_mode_f_stage1({{ARGS}}{{{{SHARED}}
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {{
        long long ordinal = start + tid;
        unsigned long long packed = pack_program(ordinal);
        atomicAdd(&sh_status[0], 1ULL);
        if (!structural_ok(packed)) {{
            atomicAdd(&sh_status[4], 1ULL);
        }} else {{
            atomicAdd(&sh_status[1], 1ULL);
            if (!has_variable(packed)) {{
                atomicAdd(&sh_status[2], 1ULL);
            }} else {{
                atomicAdd(&sh_status[5], 1ULL);
                double sums[S1_PTS];
                double terms[S1_PTS];
                double acc = 0.0;
                int slot = 0;
                int ok = 1;
                for (int n = 1; n <= S1_TERMS; ++n) {{
                    double t;
                    if (run_program(packed, (double)n, &t) != 0) {{ ok = 0; break; }}
                    acc += t;
                    if (!isfinite(acc) || fabs(acc) > VALUE_CAP) {{ ok = 0; break; }}
                    if (slot < S1_PTS && n == S1_CHECK[slot]) {{
                        sums[slot] = acc;
                        terms[slot] = t;
                        slot += 1;
                    }}
                }}
                if (!ok) {{
                    atomicAdd(&sh_status[2], 1ULL);
                }} else {{
                    atomicAdd(&sh_status[3], 1ULL);
                    double s_stab, l_stab;
                    double s_val = richardson(sums, S1_PTS, &s_stab);
                    double l_val = richardson(terms, S1_PTS, &l_stab);
                    if (isfinite(s_val) && fabs(s_val) <= VALUE_CAP &&
                        s_stab <= STABILITY1 * fmax(1.0, fabs(s_val))) {{
                        atomicAdd(&sh_status[6], 1ULL);
                        {{ACCOUNT_SERIES}}
                    }}
                    if (isfinite(l_val) && fabs(l_val) <= VALUE_CAP &&
                        l_stab <= STABILITY1 * fmax(1.0, fabs(l_val))) {{
                        atomicAdd(&sh_status[7], 1ULL);
                        {{ACCOUNT_LIMIT}}
                    }}
                }}
            }}
        }}
    }}
    __syncthreads();
    for (int i = threadIdx.x; i < NSUB * NSLOT * NWIN; i += blockDim.x) {{
        if (sh_win[i]) atomicAdd(&window_counts[i], sh_win[i]);
    }}
    for (int i = threadIdx.x; i < NSTATUS; i += blockDim.x) {{
        if (sh_status[i]) atomicAdd(&status_counts[i], sh_status[i]);
    }}
    for (int i = threadIdx.x; i < NSUB * NMAG; i += blockDim.x) {{
        if (sh_mag[i]) atomicAdd(&mag_hist[i], sh_mag[i]);
    }}
}}

extern "C" __global__ void eval_mode_f_stage2(
    const long long* __restrict__ ordinals,
    const int count,
    double* __restrict__ series_out,
    double* __restrict__ limit_out,
    unsigned char* __restrict__ flags_out)
{{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    long long ordinal = ordinals[tid];
    unsigned long long packed = pack_program(ordinal);
    series_out[tid] = 0.0;
    limit_out[tid] = 0.0;
    flags_out[tid] = 0;
    if (!structural_ok(packed) || !has_variable(packed)) return;
    double sums[S2_PTS];
    double terms[S2_PTS];
    double acc = 0.0;
    int slot = 0;
    for (int n = 1; n <= S2_TERMS; ++n) {{
        double t;
        if (run_program(packed, (double)n, &t) != 0) return;
        acc += t;
        if (!isfinite(acc) || fabs(acc) > VALUE_CAP) return;
        if (slot < S2_PTS && n == S2_CHECK[slot]) {{
            sums[slot] = acc;
            terms[slot] = t;
            slot += 1;
        }}
    }}
    double s_stab, l_stab;
    double s_val = richardson(sums, S2_PTS, &s_stab);
    double l_val = richardson(terms, S2_PTS, &l_stab);
    unsigned char flags = 0;
    if (isfinite(s_val) && fabs(s_val) <= VALUE_CAP &&
        s_stab <= STABILITY2 * fmax(1.0, fabs(s_val))) {{
        series_out[tid] = s_val;
        flags |= 1;
    }}
    if (isfinite(l_val) && fabs(l_val) <= VALUE_CAP &&
        l_stab <= STABILITY2 * fmax(1.0, fabs(l_val))) {{
        limit_out[tid] = l_val;
        flags |= 2;
    }}
    flags_out[tid] = flags;
}}

extern "C" __global__ void eval_list_mode_f_stage1(
    const long long* __restrict__ ordinals,
    const int count,
    double* __restrict__ series_out,
    double* __restrict__ limit_out,
    unsigned char* __restrict__ flags_out)
{{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    unsigned long long packed = pack_program(ordinals[tid]);
    series_out[tid] = 0.0;
    limit_out[tid] = 0.0;
    flags_out[tid] = 0;
    if (!structural_ok(packed) || !has_variable(packed)) return;
    double sums[S1_PTS];
    double terms[S1_PTS];
    double acc = 0.0;
    int slot = 0;
    for (int n = 1; n <= S1_TERMS; ++n) {{
        double t;
        if (run_program(packed, (double)n, &t) != 0) return;
        acc += t;
        if (!isfinite(acc) || fabs(acc) > VALUE_CAP) return;
        if (slot < S1_PTS && n == S1_CHECK[slot]) {{
            sums[slot] = acc;
            terms[slot] = t;
            slot += 1;
        }}
    }}
    double s_stab, l_stab;
    double s_val = richardson(sums, S1_PTS, &s_stab);
    double l_val = richardson(terms, S1_PTS, &l_stab);
    unsigned char flags = 0;
    if (isfinite(s_val) && fabs(s_val) <= VALUE_CAP &&
        s_stab <= STABILITY1 * fmax(1.0, fabs(s_val))) {{
        series_out[tid] = s_val;
        flags |= 1;
    }}
    if (isfinite(l_val) && fabs(l_val) <= VALUE_CAP &&
        l_stab <= STABILITY1 * fmax(1.0, fabs(l_val))) {{
        limit_out[tid] = l_val;
        flags |= 2;
    }}
    flags_out[tid] = flags;
}}
"""
    return (
        body.replace("{ARGS}", _KERNEL_ARGS)
        .replace("{SHARED}", _KERNEL_SHARED_SETUP)
        .replace("{ACCOUNT_SERIES}", _account_call(0, "s_val"))
        .replace("{ACCOUNT_LIMIT}", _account_call(1, "l_val"))
    )


def cuda_source(mode: str) -> str:
    """Full CUDA translation unit for one mode.  Generated from the declared alphabet."""

    submode_count = len(MODE_CONFIG[mode]["submodes"])
    parts = [_cuda_prelude(mode, submode_count), _ACCUMULATE_BODY]
    parts.append(_MODE_C_KERNEL if mode == "C" else _cuda_mode_f_kernels())
    return "".join(parts)


# ---------------------------------------------------------------------------
# CPU reference evaluator (cross-check, tests, and any machine with no CUDA device)
# ---------------------------------------------------------------------------


def _decode_batch(ordinals: np.ndarray, mode: str) -> np.ndarray:
    """Ordinals -> (n, L) token matrix, exactly the kernel's mixed-radix little-endian map."""

    config = MODE_CONFIG[mode]
    digits = np.asarray(mode_digits(mode), dtype=np.int64)
    base = len(digits)
    length = int(config["program_length"])
    rest = np.asarray(ordinals, dtype=np.int64).copy()
    tokens = np.empty((rest.shape[0], length), dtype=np.int64)
    for index in range(length):
        tokens[:, index] = digits[rest % base]
        rest //= base
    return tokens


def _structural_ok_batch(tokens: np.ndarray) -> np.ndarray:
    depth = np.zeros(tokens.shape[0], dtype=np.int64)
    alive = np.ones(tokens.shape[0], dtype=bool)
    arity_table = np.asarray(TOKEN_ARITY, dtype=np.int64)
    for index in range(tokens.shape[1]):
        arity = arity_table[tokens[:, index]]
        terminal = arity == 0
        alive &= ~(terminal & (depth >= STACK_DEPTH))
        alive &= ~(~terminal & (depth < arity))
        depth = np.where(terminal, depth + 1, depth - np.maximum(arity - 1, 0))
        depth = np.clip(depth, 0, STACK_DEPTH)
    return alive & (depth == 1)


def _apply_unary_np(op: int, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        if op == 0:
            return -value, np.ones_like(value, dtype=bool)
        if op == 1:
            ok = value != 0
            return np.divide(1.0, np.where(ok, value, 1.0)), ok
        if op == 2:
            return value * value, np.ones_like(value, dtype=bool)
        if op == 3:
            ok = value >= 0
            return np.sqrt(np.where(ok, value, 0.0)), ok
        if op == 4:
            return np.exp(value), np.ones_like(value, dtype=bool)
        if op == 5:
            ok = value > 0
            return np.log(np.where(ok, value, 1.0)), ok
        if op == 6:
            ok = np.abs(value) <= TRIG_ARGUMENT_BOUND
            return np.sin(np.where(ok, value, 0.0)), ok
        if op == 7:
            ok = np.abs(value) <= TRIG_ARGUMENT_BOUND
            return np.cos(np.where(ok, value, 0.0)), ok
        return np.arctan(value), np.ones_like(value, dtype=bool)


def _apply_binary_np(op: int, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        if op == 0:
            return left + right, np.ones_like(left, dtype=bool)
        if op == 1:
            return left - right, np.ones_like(left, dtype=bool)
        if op == 2:
            return left * right, np.ones_like(left, dtype=bool)
        if op == 3:
            ok = right != 0
            return np.divide(left, np.where(ok, right, 1.0)), ok
        ok = ~((left < 0) & (right != np.trunc(right))) & ~((left == 0) & (right <= 0))
        base = np.where(ok, left, 1.0)
        return np.power(base, right), ok


def evaluate_batch_cpu(
    ordinals: np.ndarray, mode: str, variable: np.ndarray | float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised numpy stack machine: the same semantics as the kernel, on the CPU.

    Returns ``(values, usable)``.  ``usable`` is false for structurally invalid programs and
    for any program that left the declared domain or exceeded the magnitude cap.
    """

    tokens = _decode_batch(ordinals, mode)
    count = tokens.shape[0]
    usable = _structural_ok_batch(tokens)
    stack = np.zeros((count, STACK_DEPTH), dtype=np.float64)
    pointer = np.zeros(count, dtype=np.int64)
    rows = np.arange(count)
    variable_array = (
        None if variable is None else np.broadcast_to(np.asarray(variable, dtype=np.float64), count)
    )
    for index in range(tokens.shape[1]):
        column = tokens[:, index]
        for token in range(TOKEN_COUNT):
            selected = usable & (column == token)
            if not selected.any():
                continue
            arity = TOKEN_ARITY[token]
            if arity == 0:
                if token == VARIABLE_INDEX:
                    if variable_array is None:
                        usable &= ~selected
                        continue
                    pushed = variable_array[selected]
                else:
                    pushed = np.full(int(selected.sum()), TERMINAL_FLOATS[token])
                stack[rows[selected], pointer[selected]] = pushed
                pointer[selected] += 1
                continue
            if arity == 1:
                operand = stack[rows[selected], pointer[selected] - 1]
                value, ok = _apply_unary_np(token - UNARY_START, operand)
                good = ok & np.isfinite(value) & (np.abs(value) <= VALUE_CAP)
                stack[rows[selected], pointer[selected] - 1] = np.where(good, value, 0.0)
            else:
                left = stack[rows[selected], pointer[selected] - 2]
                right = stack[rows[selected], pointer[selected] - 1]
                value, ok = _apply_binary_np(token - BINARY_START, left, right)
                good = ok & np.isfinite(value) & (np.abs(value) <= VALUE_CAP)
                pointer[selected] -= 1
                stack[rows[selected], pointer[selected] - 1] = np.where(good, value, 0.0)
            failed = np.zeros(count, dtype=bool)
            failed[np.nonzero(selected)[0][~good]] = True
            usable &= ~failed
    values = stack[:, 0]
    return np.where(usable, values, 0.0), usable


def evaluate_series_batch_cpu(
    ordinals: np.ndarray,
    mode: str,
    terms: int,
    ladder: int,
    stability: float = STAGE1_STABILITY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU twin of the Mode F kernels: Richardson-extrapolated series value and limit."""

    checkpoints = _checkpoints(terms, ladder)
    count = np.asarray(ordinals).shape[0]
    tokens = _decode_batch(np.asarray(ordinals), mode)
    alive = _structural_ok_batch(tokens) & (tokens == VARIABLE_INDEX).any(axis=1)
    accumulator = np.zeros(count, dtype=np.float64)
    sums = np.zeros((count, ladder), dtype=np.float64)
    terms_at = np.zeros((count, ladder), dtype=np.float64)
    slot = 0
    for step in range(1, terms + 1):
        value, usable = evaluate_batch_cpu(np.asarray(ordinals), mode, float(step))
        alive &= usable
        accumulator = accumulator + np.where(alive, value, 0.0)
        alive &= np.isfinite(accumulator) & (np.abs(accumulator) <= VALUE_CAP)
        if slot < ladder and step == checkpoints[slot]:
            sums[:, slot] = accumulator
            terms_at[:, slot] = value
            slot += 1
    series_value, series_ok = _richardson_np(sums, stability)
    limit_value, limit_ok = _richardson_np(terms_at, stability)
    return (
        np.where(alive & series_ok, series_value, 0.0),
        np.where(alive & limit_ok, limit_value, 0.0),
        np.stack([alive & series_ok, alive & limit_ok], axis=1),
    )


def _richardson_np(
    table: np.ndarray, stability: float = STAGE2_STABILITY
) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise Richardson extrapolation with the same stability test as the kernel."""

    work = table.copy()
    points = work.shape[1]
    previous = work[:, points - 1].copy()
    for level in range(1, points):
        factor = float(1 << level)
        for index in range(points - level):
            work[:, index] = (factor * work[:, index + 1] - work[:, index]) / (factor - 1.0)
        if level == points - 2:
            previous = work[:, 1].copy()
    value = work[:, 0]
    residual = np.abs(value - previous)
    ok = (
        np.isfinite(value)
        & (np.abs(value) <= VALUE_CAP)
        & (residual <= stability * np.maximum(1.0, np.abs(value)))
    )
    return value, ok


# ---------------------------------------------------------------------------
# HyperLogLog: distinct evaluated values, measured with a declared standard error
# ---------------------------------------------------------------------------


def hll_estimate(registers: np.ndarray) -> float:
    """Cardinality estimate from HyperLogLog registers, with the small-range correction."""

    count = int(registers.size)
    alpha = 0.7213 / (1.0 + 1.079 / count)
    harmonic = float(np.sum(np.exp2(-registers.astype(np.float64))))
    if harmonic == 0.0:
        return 0.0
    estimate = alpha * count * count / harmonic
    zeros = int(np.count_nonzero(registers == 0))
    if estimate <= 2.5 * count and zeros > 0:
        estimate = count * math.log(count / zeros)
    return float(estimate)


HLL_STANDARD_ERROR = 1.04 / math.sqrt(HLL_REGISTERS)


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def hll_registers_from_hashes(hashes: np.ndarray) -> np.ndarray:
    """Host twin of the kernel's HyperLogLog update: index from the top bits, rank from the
    leading zeros of the rest, with the same sentinel bit that bounds the rank."""

    registers = np.zeros(HLL_REGISTERS, dtype=np.uint32)
    if hashes.size == 0:
        return registers
    values = np.asarray(hashes, dtype=np.uint64)
    index = (values >> np.uint64(64 - HLL_PRECISION)).astype(np.int64)
    remainder = (values << np.uint64(HLL_PRECISION)) | np.uint64(1 << (HLL_PRECISION - 1))
    rank = np.full(values.size, 64 - HLL_PRECISION + 1, dtype=np.uint32)
    found = np.zeros(values.size, dtype=bool)
    for position in range(64):
        bit = (remainder >> np.uint64(63 - position)) & np.uint64(1)
        fresh = (~found) & (bit == 1)
        rank[fresh] = position + 1
        found |= fresh
        if found.all():
            break
    np.maximum.at(registers, index, rank)
    return registers


def value_hash_keys(values: np.ndarray) -> np.ndarray:
    """Host twin of the kernel's quantise-then-hash, for the exact-distinct cross-check."""

    mantissa, exponent = np.frexp(values)
    scaled = np.rint(mantissa * float(1 << VALUE_HASH_MANTISSA_BITS)).astype(np.int64)
    encoded_exponent = ((exponent.astype(np.int64) + 4096) & 0x1FFF).astype(np.uint64)
    encoded_mantissa = ((scaled + (1 << 41)) & 0x3FFFFFFFFFF).astype(np.uint64)
    return (encoded_exponent << np.uint64(44)) | encoded_mantissa


# ---------------------------------------------------------------------------
# GPU engine
# ---------------------------------------------------------------------------


class SweepParameters:
    """Per-(submode, slot) retention windows and sampling thresholds for one sweep."""

    __slots__ = ("threshold_band", "threshold_core", "window_band", "window_core")

    def __init__(self, submodes: int) -> None:
        size = submodes * SLOT_COUNT
        self.window_core = np.zeros(size, dtype=np.float64)
        self.threshold_core = np.zeros(size, dtype=np.uint64)
        self.window_band = np.zeros(size, dtype=np.float64)
        self.threshold_band = np.zeros(size, dtype=np.uint64)

    @staticmethod
    def rate_to_threshold(rate: float) -> int:
        if rate >= 1.0:
            return (1 << 64) - 1
        if rate <= 0.0:
            return 0
        return int(rate * float(1 << 64))

    def as_json(self, submodes: Sequence[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, name in enumerate(submodes):
            for slot in TARGET_SLOTS:
                flat = index * SLOT_COUNT + int(slot["slot"])
                rows.append(
                    {
                        "submode": name,
                        "target": slot["name"],
                        "core_window": format(float(self.window_core[flat]), ".3e"),
                        "core_rate": format(
                            float(self.threshold_core[flat]) / float(1 << 64), ".6e"
                        ),
                        "band_window": format(float(self.window_band[flat]), ".3e"),
                        "band_rate": format(
                            float(self.threshold_band[flat]) / float(1 << 64), ".6e"
                        ),
                    }
                )
        return rows


class GpuUnavailableError(CompositionalSearchError):
    """Raised when a GPU sweep is requested on a machine with no working CUDA device."""


class _Engine:
    """Compiled kernels plus the device-side accumulators for one mode."""

    def __init__(self, mode: str, threads: int = DEFAULT_THREADS_PER_BLOCK) -> None:
        try:
            import cupy as cp
        except Exception as error:  # any CUDA absence is the same condition
            raise GpuUnavailableError(f"cupy is unavailable: {error}") from error
        self.cp = cp
        self.mode = mode
        self.threads = threads
        self.submodes = tuple(MODE_CONFIG[mode]["submodes"])
        self.n_sub = len(self.submodes)
        self.module = cp.RawModule(
            code=cuda_source(mode), options=("-std=c++14",), backend="nvrtc"
        )
        self.sweep_kernel = self.module.get_function(
            "eval_mode_c" if mode == "C" else "eval_mode_f_stage1"
        )
        self.list_kernel = self.module.get_function(
            "eval_list_mode_c" if mode == "C" else "eval_list_mode_f_stage1"
        )
        self.stage2_kernel = (
            None if mode == "C" else self.module.get_function("eval_mode_f_stage2")
        )
        self.targets = cp.asarray(SLOT_VALUES)
        self.scales = cp.asarray(1.0 / np.maximum(1.0, np.abs(SLOT_VALUES)))
        self.window_counts = cp.zeros(self.n_sub * SLOT_COUNT * len(WINDOWS), dtype=cp.uint64)
        self.status_counts = cp.zeros(NSTATUS, dtype=cp.uint64)
        self.mag_hist = cp.zeros(self.n_sub * NMAG, dtype=cp.uint64)
        self.hll = cp.zeros(self.n_sub * HLL_REGISTERS, dtype=cp.uint32)
        self.hit_ordinal = cp.zeros(HIT_CAPACITY, dtype=cp.int64)
        self.hit_slot = cp.zeros(HIT_CAPACITY, dtype=cp.int32)
        self.hit_value = cp.zeros(HIT_CAPACITY, dtype=cp.float64)
        self.hit_count = cp.zeros(1, dtype=cp.int32)
        self.device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

    def reset(self) -> None:
        self.window_counts[:] = 0
        self.status_counts[:] = 0
        self.mag_hist[:] = 0
        self.hll[:] = 0

    def load_state(self, state: Mapping[str, Any]) -> None:
        cp = self.cp
        self.window_counts = cp.asarray(_decode_array(state["window_counts"], np.uint64))
        self.status_counts = cp.asarray(_decode_array(state["status_counts"], np.uint64))
        self.mag_hist = cp.asarray(_decode_array(state["mag_hist"], np.uint64))
        self.hll = cp.asarray(_decode_array(state["hll"], np.uint32))

    def dump_state(self) -> dict[str, str]:
        return {
            "window_counts": _encode_array(self.window_counts.get()),
            "status_counts": _encode_array(self.status_counts.get()),
            "mag_hist": _encode_array(self.mag_hist.get()),
            "hll": _encode_array(self.hll.get()),
        }

    def launch(
        self, start: int, count: int, parameters: SweepParameters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One kernel launch.  On retention-buffer overflow the launch splits and re-runs."""

        cp = self.cp
        device_parameters = (
            cp.asarray(parameters.window_core),
            cp.asarray(parameters.threshold_core),
            cp.asarray(parameters.window_band),
            cp.asarray(parameters.threshold_band),
        )
        return self._launch_recursive(start, count, device_parameters, depth=0)

    def _launch_recursive(
        self, start: int, count: int, device_parameters: tuple[Any, ...], depth: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cp = self.cp
        if count <= 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
        if depth > 40:
            raise CompositionalSearchError("retention buffer overflow could not be split away")
        before = self.window_counts.copy(), self.status_counts.copy(), self.mag_hist.copy()
        self.hit_count[:] = 0
        blocks = (count + self.threads - 1) // self.threads
        self.sweep_kernel(
            (blocks,),
            (self.threads,),
            (
                np.int64(start),
                np.int32(count),
                self.targets,
                self.scales,
                self.window_counts,
                self.status_counts,
                self.mag_hist,
                self.hll,
                *device_parameters,
                self.hit_ordinal,
                self.hit_slot,
                self.hit_value,
                self.hit_count,
                np.int32(HIT_CAPACITY),
            ),
        )
        found = int(self.hit_count.get()[0])
        if found > HIT_CAPACITY:
            # Roll the accumulators back and redo the launch in halves; nothing is lost and
            # the retained set stays a pure function of the ordinal.
            self.window_counts[:] = before[0]
            self.status_counts[:] = before[1]
            self.mag_hist[:] = before[2]
            half = count // 2
            left = self._launch_recursive(start, half, device_parameters, depth + 1)
            right = self._launch_recursive(start + half, count - half, device_parameters, depth + 1)
            return (
                np.concatenate([left[0], right[0]]),
                np.concatenate([left[1], right[1]]),
                np.concatenate([left[2], right[2]]),
            )
        if found == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64)
        ordinals = cp.asnumpy(self.hit_ordinal[:found])
        slots = cp.asnumpy(self.hit_slot[:found])
        values = cp.asnumpy(self.hit_value[:found])
        order = np.lexsort((slots, ordinals))
        return ordinals[order], slots[order], values[order]

    def evaluate_list(self, ordinals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """List-driven evaluation used by the CPU/GPU cross-check."""

        cp = self.cp
        count = int(ordinals.size)
        device_ordinals = cp.asarray(np.asarray(ordinals, dtype=np.int64))
        if self.mode == "C":
            values = cp.zeros(count, dtype=cp.float64)
            flags = cp.zeros(count, dtype=cp.uint8)
            blocks = (count + self.threads - 1) // self.threads
            self.list_kernel(
                (blocks,), (self.threads,), (device_ordinals, np.int32(count), values, flags)
            )
            return cp.asnumpy(values), cp.asnumpy(flags)
        series, limit, flags = self._run_pair(self.list_kernel, device_ordinals, count)
        return np.stack([series, limit], axis=1), flags

    def stage2(self, ordinals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mode F stage 2: the near-target set re-summed at 64x the stage-1 depth."""

        if self.stage2_kernel is None:
            raise CompositionalSearchError("stage 2 exists only for mode F")
        cp = self.cp
        count = int(ordinals.size)
        if count == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, np.empty(0, dtype=np.uint8)
        device_ordinals = cp.asarray(np.asarray(ordinals, dtype=np.int64))
        return self._run_pair(self.stage2_kernel, device_ordinals, count)

    def _run_pair(
        self, kernel: Any, device_ordinals: Any, count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cp = self.cp
        series = cp.zeros(count, dtype=cp.float64)
        limit = cp.zeros(count, dtype=cp.float64)
        flags = cp.zeros(count, dtype=cp.uint8)
        blocks = (count + self.threads - 1) // self.threads
        kernel((blocks,), (self.threads,), (device_ordinals, np.int32(count), series, limit, flags))
        return cp.asnumpy(series), cp.asnumpy(limit), cp.asnumpy(flags)


def _encode_array(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _decode_array(text: str, dtype: Any) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text.encode("ascii")), dtype=dtype).copy()


# ---------------------------------------------------------------------------
# Chunked, restart-safe sweeps
# ---------------------------------------------------------------------------


def _merge_blocks(
    blocks: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate retained-hit blocks, keeping the three parallel arrays aligned."""

    if not blocks:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.concatenate([block[0] for block in blocks]),
        np.concatenate([block[1] for block in blocks]),
        np.concatenate([block[2] for block in blocks]),
    )


def _chunk_lineage(previous: str, summary: Mapping[str, Any]) -> str:
    return hashlib.sha256((previous + canonical_sha256(summary)).encode("ascii")).hexdigest()


def run_sweep(
    engine: _Engine,
    *,
    pass_name: str,
    total: int,
    parameters: SweepParameters,
    chunk_size: int,
    launch_size: int,
    checkpoint_path: Path | None,
    hit_budget: int,
) -> dict[str, Any]:
    """Enumerate ``[0, total)`` in chunks, checkpointing after each one.

    The checkpoint carries every accumulator plus the next ordinal and the rolling chunk
    lineage hash, so an interrupted campaign resumes at chunk granularity and produces the
    same receipt it would have produced without the interruption.
    """

    state: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "mode": engine.mode,
        "pass": pass_name,
        "total": total,
        "chunk_size": chunk_size,
        "launch_size": launch_size,
        "next_ordinal": 0,
        "chunks": 0,
        "lineage": hashlib.sha256(f"{engine.mode}|{pass_name}|{total}".encode()).hexdigest(),
        "elapsed_seconds": "0.000000",
        "hits": {"ordinal": "", "slot": "", "value": ""},
        "hits_truncated": False,
    }
    if checkpoint_path is not None and checkpoint_path.exists():
        loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            loaded.get("schema_version") == CHECKPOINT_SCHEMA
            and loaded.get("mode") == engine.mode
            and loaded.get("pass") == pass_name
            and loaded.get("total") == total
            and loaded.get("chunk_size") == chunk_size
        ):
            state.update(
                {key: loaded[key] for key in ("next_ordinal", "chunks", "lineage", "hits")}
            )
            # Sweep time accumulates across resumes.  A rate computed from one session's
            # wall clock after a resume would be the ratio of all the work to some of the
            # time, which is not a measurement of anything.
            state["elapsed_seconds"] = str(loaded.get("elapsed_seconds", "0.000000"))
            state["hits_truncated"] = bool(loaded.get("hits_truncated", False))
            engine.load_state(loaded["accumulators"])

    started = time.perf_counter()
    resumed_from = int(state["next_ordinal"])
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if state["hits"]["ordinal"]:
        blocks.append(
            (
                _decode_array(state["hits"]["ordinal"], np.int64),
                _decode_array(state["hits"]["slot"], np.int32),
                _decode_array(state["hits"]["value"], np.float64),
            )
        )
    retained = int(blocks[0][0].size) if blocks else 0
    while state["next_ordinal"] < total:
        chunk_start = int(state["next_ordinal"])
        chunk_stop = min(chunk_start + chunk_size, total)
        for launch_start in range(chunk_start, chunk_stop, launch_size):
            launch_stop = min(launch_start + launch_size, chunk_stop)
            found = engine.launch(launch_start, launch_stop - launch_start, parameters)
            if found[0].size and not state["hits_truncated"]:
                keep = min(hit_budget - retained, int(found[0].size))
                if keep < int(found[0].size):
                    state["hits_truncated"] = True
                if keep > 0:
                    blocks.append((found[0][:keep], found[1][:keep], found[2][:keep]))
                    retained += keep
        state["next_ordinal"] = chunk_stop
        state["chunks"] = int(state["chunks"]) + 1
        state["lineage"] = _chunk_lineage(
            str(state["lineage"]),
            {"chunk": int(state["chunks"]), "start": chunk_start, "stop": chunk_stop},
        )
        state["elapsed_seconds"] = format(
            float(state["elapsed_seconds"]) + (time.perf_counter() - started), ".6f"
        )
        started = time.perf_counter()
        if checkpoint_path is not None:
            merged = _merge_blocks(blocks)
            blocks = [merged]
            state["hits"] = {
                "ordinal": _encode_array(merged[0]),
                "slot": _encode_array(merged[1]),
                "value": _encode_array(merged[2]),
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {**state, "accumulators": engine.dump_state()}
            checkpoint_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    hits = _merge_blocks(blocks)
    elapsed = float(state["elapsed_seconds"])

    window_counts = engine.window_counts.get().reshape(engine.n_sub, SLOT_COUNT, len(WINDOWS))
    return {
        "pass": pass_name,
        "mode": engine.mode,
        "enumerated": total,
        "chunks": int(state["chunks"]),
        "chunk_size": chunk_size,
        "launch_size": launch_size,
        "chunk_lineage_sha256": str(state["lineage"]),
        "elapsed_seconds": elapsed,
        "resumed_from_ordinal": resumed_from,
        "swept_this_session": total - resumed_from,
        "window_counts": window_counts,
        "status_counts": engine.status_counts.get(),
        "mag_hist": engine.mag_hist.get().reshape(engine.n_sub, NMAG),
        "hll": engine.hll.get().reshape(engine.n_sub, HLL_REGISTERS),
        "hits": hits,
        "hits_truncated": bool(state["hits_truncated"]),
    }


# ---------------------------------------------------------------------------
# THE CHANCE-MATCH GATE
# ---------------------------------------------------------------------------


def _density_scaling(cumulative: np.ndarray) -> dict[str, Any]:
    """Measured exponent of ``C(W) ~ W^p`` over the windows the density model extrapolates.

    The chance model extrapolates a density measured at 1e-3-ish down to the match window, and
    that extrapolation is only as good as the assumption ``C(W)`` is linear in ``W``.  This is
    the check: a least-squares slope of ``log C`` against ``log W`` over the windows carrying
    enough counts.  A slope near 1 supports the extrapolation; a slope far from 1 is a warning
    written into the receipt rather than an assumption made quietly.
    """

    xs: list[float] = []
    ys: list[float] = []
    for index, exponent in enumerate(WINDOW_EXPONENTS):
        if exponent < DENSITY_MAX_EXPONENT:
            continue
        population = int(cumulative[index])
        if population < DENSITY_MIN_COUNT:
            continue
        xs.append(-float(exponent))
        ys.append(math.log10(population))
    if len(xs) < 3:
        return {"points": len(xs), "slope": None, "supports_linear_extrapolation": None}
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return {"points": len(xs), "slope": None, "supports_linear_extrapolation": None}
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
    return {
        "points": len(xs),
        "slope": format(slope, ".4f"),
        "supports_linear_extrapolation": bool(0.6 <= slope <= 1.4),
        "note": "C(W) ~ W^slope over the fitted windows; slope 1 is the linear-density model",
    }


def chance_accounting(counts: np.ndarray, submodes: Sequence[str]) -> dict[str, Any]:
    """Per-target chance model read off the measured value distribution.

    ``counts`` is the cumulative window census ``C[sub, slot, w]``: how many evaluated values
    landed within ``WINDOWS[w]`` of that target in scaled distance.  The local density is taken
    from an *annulus* ``[W[j+1], W[j])`` rather than from a nested window, so an expressible
    target's exact-match spike -- which sits at distance ~1e-16 and is inside every nested
    window -- cannot inflate the chance model that is supposed to be measuring the noise
    around it.  The expected number of chance matches at tolerance ``eps`` is then
    ``2 * eps * density``, and the tolerance at which that expectation drops below the declared
    threshold is the precision this run demands of a survivor.
    """

    rows: list[dict[str, Any]] = []
    for sub_index, submode in enumerate(submodes):
        for slot in TARGET_SLOTS:
            slot_index = int(slot["slot"])
            cumulative = counts[sub_index, slot_index].astype(np.int64)
            annuli: list[dict[str, Any]] = []
            density: float | None = None
            density_source: dict[str, Any] | None = None
            for window_index in range(len(WINDOWS) - 1):
                inner = float(WINDOWS[window_index + 1])
                outer = float(WINDOWS[window_index])
                population = int(cumulative[window_index] - cumulative[window_index + 1])
                measure = 2.0 * (outer - inner)
                entry = {
                    "annulus": f"[{inner:.0e}, {outer:.0e})",
                    "count": population,
                    "density_per_unit": format(population / measure, ".6e"),
                }
                annuli.append(entry)
                if (
                    density is None
                    and WINDOW_EXPONENTS[window_index] >= DENSITY_MAX_EXPONENT
                    and population >= DENSITY_MIN_COUNT
                ):
                    density = population / measure
                    density_source = {
                        "annulus": entry["annulus"],
                        "count": population,
                        "rule": (
                            f"tightest annulus at or inside 1e-{DENSITY_MAX_EXPONENT} holding "
                            f"at least {DENSITY_MIN_COUNT} values"
                        ),
                    }
            if density is None:
                # Nothing was dense enough to measure.  Bound the density from above by
                # "at most one value in the tightest annulus considered" and say so.
                inner = float(WINDOWS[-1])
                outer = float(WINDOWS[-2])
                density = 1.0 / (2.0 * (outer - inner))
                density_source = {
                    "annulus": f"[{inner:.0e}, {outer:.0e})",
                    "count": 0,
                    "rule": (
                        "no annulus reached the minimum count; density bounded above by one "
                        "value in the tightest annulus (conservative)"
                    ),
                }
            match_index = WINDOW_EXPONENTS.index(MATCH_WINDOW_EXPONENT)
            observed = int(cumulative[match_index])
            expected = 2.0 * MATCH_WINDOW * density
            tolerance = CHANCE_THRESHOLD / (2.0 * density)
            required_digits = max(1, math.ceil(-math.log10(tolerance)))
            scaling = _density_scaling(cumulative)
            rows.append(
                {
                    "submode": submode,
                    "target": slot["name"],
                    "role": slot["role"],
                    "paired_with": slot["paired_with"],
                    "target_value": format(float(slot["value"]), ".17g"),
                    "cumulative_counts": {
                        f"1e-{exponent}": int(cumulative[index])
                        for index, exponent in enumerate(WINDOW_EXPONENTS)
                    },
                    "annuli": annuli,
                    "local_density_per_unit": format(density, ".6e"),
                    "density_source": density_source,
                    "fp64_matches_at_1e-12": observed,
                    "expected_chance_matches": format(expected, ".6e"),
                    "excess_over_chance": format(observed - expected, ".6e"),
                    "tolerance_for_chance_below_threshold": format(tolerance, ".6e"),
                    "required_agreement_digits": required_digits,
                    "count_scaling_exponent": scaling,
                }
            )
    required = max(int(row["required_agreement_digits"]) for row in rows) if rows else 1
    verification_dps = max(VERIFY_DPS_FLOOR, required + VERIFY_GUARD_DIGITS)
    return {
        "threshold_expected_chance_matches": format(CHANCE_THRESHOLD, ".0e"),
        "distance_measure": "|value - target| / max(1, |target|)",
        "match_window": format(MATCH_WINDOW, ".0e"),
        "density_rule": (
            "local density from the tightest annulus at or inside "
            f"1e-{DENSITY_MAX_EXPONENT} holding at least {DENSITY_MIN_COUNT} values; annuli "
            "are used rather than nested windows so an exact-match spike at the target cannot "
            "inflate its own chance model"
        ),
        "derivation": (
            "expected_chance_matches(eps) = 2 * eps * local_density; solving "
            "2 * eps * local_density = threshold gives the per-target tolerance, and "
            "required_agreement_digits = ceil(-log10(tolerance)); the campaign's verification "
            "precision P is the largest per-target requirement plus "
            f"{VERIFY_GUARD_DIGITS} guard digits, floored at {VERIFY_DPS_FLOOR}"
        ),
        "note_when_requirement_is_at_or_below_the_match_window": (
            "a derived requirement at or below the match-window exponent means the measured "
            "density already puts the expected chance count below the threshold at fp64; the "
            "decoy column is what confirms that reading empirically"
        ),
        "max_required_agreement_digits": required,
        "verification_dps": verification_dps,
        "verification_dps_floor": VERIFY_DPS_FLOOR,
        "guard_digits": VERIFY_GUARD_DIGITS,
        "per_target": rows,
    }


def derive_retention(
    census: Mapping[str, Any], submodes: Sequence[str], mode: str
) -> SweepParameters:
    """Turn the census counts into per-slot retention windows and sampling rates."""

    counts = census["window_counts"]
    parameters = SweepParameters(len(submodes))
    if mode == "F":
        near_index = WINDOW_EXPONENTS.index(round(-math.log10(MODE_F_NEAR_WINDOW)))
        for sub_index in range(len(submodes)):
            for slot in TARGET_SLOTS:
                flat = sub_index * SLOT_COUNT + int(slot["slot"])
                population = int(counts[sub_index, int(slot["slot"]), near_index])
                rate = 1.0 if population <= RETENTION_BUDGET_MODE_F_NEAR else (
                    RETENTION_BUDGET_MODE_F_NEAR / population
                )
                parameters.window_core[flat] = MODE_F_NEAR_WINDOW
                parameters.threshold_core[flat] = SweepParameters.rate_to_threshold(rate)
        return parameters
    core_index = WINDOW_EXPONENTS.index(CORE_WINDOW_EXPONENT)
    band_index = WINDOW_EXPONENTS.index(STRESS_WINDOW_EXPONENT)
    for sub_index in range(len(submodes)):
        for slot in TARGET_SLOTS:
            slot_index = int(slot["slot"])
            flat = sub_index * SLOT_COUNT + slot_index
            core_population = int(counts[sub_index, slot_index, core_index])
            # The stress tier samples the annulus, so its rate is set by the annulus
            # population; sampling the nested window instead would spend the whole budget on
            # the exact-identity atom and never test the gate on a chance match at all.
            annulus = max(
                0, int(counts[sub_index, slot_index, band_index]) - core_population
            )
            core_rate = 1.0 if core_population <= RETENTION_BUDGET_CORE else (
                RETENTION_BUDGET_CORE / core_population
            )
            band_rate = 1.0 if annulus <= RETENTION_BUDGET_BAND else (
                RETENTION_BUDGET_BAND / annulus
            )
            parameters.window_core[flat] = CORE_WINDOW
            parameters.threshold_core[flat] = SweepParameters.rate_to_threshold(core_rate)
            parameters.window_band[flat] = STRESS_WINDOW
            parameters.threshold_band[flat] = SweepParameters.rate_to_threshold(band_rate)
    return parameters


# ---------------------------------------------------------------------------
# Verification at the derived precision
# ---------------------------------------------------------------------------

#: Per (submode, slot, tier) cap on how many retained candidates are re-verified in mpmath.
#: Selection is closest-first by fp64 distance with an ordinal tie-break, so it is a declared
#: deterministic subset and never a race.
VERIFY_BUDGET_PER_SLOT = 64


def verify_candidate(
    ordinal: int,
    mode: str,
    submode: str,
    slot: Mapping[str, Any],
    required_digits: int,
    dps: int,
) -> dict[str, Any]:
    """Re-evaluate one candidate at the derived precision and again at twice that precision.

    Two bars, both of which must hold at both precisions.  The first is the chance bar:
    ``required_digits``, derived from the measured local value density, is how many digits it
    takes to beat coincidence at this campaign size.  The second is the holdout bar: agreement
    within ``AGREEMENT_GUARD_DIGITS`` of the working precision, which is what separates an
    identity (whose agreement grows with the precision) from an asymptotic approximation
    (whose agreement is fixed by its own error term and stops growing).  Neither bar alone is
    enough -- the chance bar misses saturating functions, and the holdout bar alone would not
    know how much precision this campaign size demands.
    """

    tokens = decode_ordinal(ordinal, mode)
    report: dict[str, Any] = {
        "ordinal": ordinal,
        "mode": mode,
        "submode": submode,
        "target": slot["name"],
        "target_role": slot["role"],
        "rpn": render_rpn(tokens),
        "infix": render_infix(tokens),
        "required_agreement_digits": required_digits,
    }
    stages: list[dict[str, Any]] = []
    survived = True
    value_text = None
    for stage_dps in (dps, 2 * dps):
        holdout_digits = stage_dps - AGREEMENT_GUARD_DIGITS
        bar = max(required_digits, holdout_digits)
        with mp.workdps(stage_dps):
            value = evaluate_program_value_mp(tokens, submode)
            if value is None:
                stages.append({"dps": stage_dps, "status": "UNDEFINED"})
                survived = False
                break
            target = slot_value_mp(slot)
            scale = max(mp.mpf(1), abs(target))
            distance = abs(value - target) / scale
            agreement = (
                stage_dps if distance == 0 else int(mp.floor(-mp.log10(distance)))
            )
            if stage_dps == dps:
                value_text = mp.nstr(value, min(stage_dps - 5, 60))
            failure = (
                None
                if agreement >= bar
                else (
                    "CHANCE_BAR"
                    if agreement < required_digits
                    else "HOLDOUT_BAR_AGREEMENT_DID_NOT_TRACK_PRECISION"
                )
            )
            stages.append(
                {
                    "dps": stage_dps,
                    "status": "AGREES" if failure is None else "DIVERGES",
                    "agreement_digits": int(agreement),
                    "chance_bar_digits": required_digits,
                    "holdout_bar_digits": holdout_digits,
                    "failed_bar": failure,
                    "scaled_distance": mp.nstr(distance, 6),
                }
            )
            if failure is not None:
                survived = False
                break
    report["verification_stages"] = stages
    report["survived_chance_gate"] = survived
    report["value"] = value_text
    return report


# ---------------------------------------------------------------------------
# Downstream: prior-art screening and classical reduction (reused, not reinvented)
# ---------------------------------------------------------------------------

CORPUS_DATABASE = Path("runs/math/prior-art/cf-corpus-v1.sqlite")
CORPUS_MANIFEST = Path("runs/math/prior-art/cf-corpus-v1-manifest.json")

#: Classical identities that live inside this grammar's own space.  Rediscovering them is the
#: control that proves the enumerator reaches real mathematics; the table is finite and
#: explicit, and absence from it is never novelty.
BUILTIN_KNOWN_IDENTITIES: tuple[dict[str, Any], ...] = (
    {
        "id": "pi_gregory_leibniz_point",
        "mode": "C",
        "submode": "constant",
        "target": "pi",
        "rpn": "1 atan 2 sqr mul 1 mul 1 mul",
        "attribution": "pi = 4 arctan 1 (Gregory-Leibniz / Machin family base point)",
    },
    {
        "id": "pi_euler_machin_like",
        "mode": "C",
        "submode": "constant",
        "target": "pi",
        "rpn": "1/2 atan 1/3 atan add 4 mul 1 mul",
        "attribution": "Euler 1738: pi/4 = arctan(1/2) + arctan(1/3), a Machin-like formula",
    },
    {
        "id": "phi_from_sqrt5",
        "mode": "C",
        "submode": "constant",
        "target": "phi",
        "rpn": "1 5 sqrt add 2 recip mul 1 mul",
        "attribution": "golden ratio phi = (1 + sqrt 5)/2 (Euclid, Elements VI.30)",
    },
    {
        "id": "e_from_exp_half",
        "mode": "C",
        "submode": "constant",
        "target": "e",
        "rpn": "1/2 exp sqr 1 mul 1 mul 1 mul",
        "attribution": "e = exp(1/2)^2, the exponential at a rational argument",
    },
    {
        "id": "zeta2_basel_closed_form",
        "mode": "C",
        "submode": "constant",
        "target": "zeta2",
        "rpn": "1 atan 4 mul sqr 2 3 mul div",
        "attribution": "Euler 1735 (Basel problem): zeta(2) = pi^2/6",
    },
    {
        "id": "ln2_from_sqrt4",
        "mode": "C",
        "submode": "constant",
        "target": "ln2",
        "rpn": "4 sqrt ln 1 mul 1 mul 1 mul",
        "attribution": "ln 2 = ln sqrt 4, the logarithm at a rational argument",
    },
    {
        "id": "sqrt2_from_sqrt_sqrt4",
        "mode": "C",
        "submode": "constant",
        "target": "sqrt2",
        "rpn": "4 sqrt sqrt 1 mul 1 mul 1 mul",
        "attribution": "sqrt 2 = sqrt sqrt 4 (Pythagoras / Euclid X)",
    },
    {
        "id": "e_pi_gelfond",
        "mode": "C",
        "submode": "constant",
        "target": "e_pi",
        "rpn": "1 atan 4 mul exp 1 mul 1 mul",
        "attribution": "Gelfond's constant e^pi = exp(4 arctan 1)",
    },
    {
        "id": "zeta2_basel_series",
        "mode": "F",
        "submode": "series",
        "target": "zeta2",
        "rpn": "k sqr recip 2 sqr 4 div mul",
        "attribution": "Euler 1735 (Basel problem): Sum_{k>=1} 1/k^2 = pi^2/6",
    },
    {
        "id": "zeta3_apery_series",
        "mode": "F",
        "submode": "series",
        "target": "zeta3",
        "rpn": "k 3 pow recip 1 mul 1 mul",
        "attribution": "Apery's constant: Sum_{k>=1} 1/k^3 = zeta(3)",
    },
    {
        "id": "e_bernoulli_limit",
        "mode": "F",
        "submode": "limit",
        "target": "e",
        "rpn": "k 1 sqr add k div k pow",
        "attribution": "Bernoulli 1683: e = lim_{k->inf} (1 + 1/k)^k",
    },
)


def _tokens_from_rpn(text: str) -> tuple[int, ...]:
    index = {name: position for position, name in enumerate(TOKEN_NAMES)}
    return tuple(index[name] for name in text.split())


def load_prior_art_values(root: Path) -> list[dict[str, Any]] | None:
    """Every corpus record's constant, for the value screen.  ``None`` if the corpus is absent."""

    database = root / CORPUS_DATABASE
    manifest = root / CORPUS_MANIFEST
    if not database.exists() or not manifest.exists():
        return None
    from .cf_prior_art_corpus import load_corpus

    corpus = load_corpus(database, manifest)
    rows: list[dict[str, Any]] = []
    for record in corpus.records:
        rows.append(
            {
                "record_id": record.record_id,
                "family": record.family,
                "value": record.value,
                "value_expr": record.value_expr,
                "citation": record.citation.as_json(),
            }
        )
    return rows


def screen_prior_art(
    value_text: str, corpus_rows: Sequence[Mapping[str, Any]] | None, digits: int = 40
) -> dict[str, Any]:
    """Adjudicate a survivor's constant against the corpus, using the repository's own rule.

    Value equality alone is never membership: two different expressions can converge to the
    same constant, and calling that "known" would turn "pi is a known number" into "this
    formula is known".  A value hit is therefore reported as ``CONSTANT_ATTESTED_IN_CORPUS``,
    which is evidence about the constant and not about the expression.
    """

    if corpus_rows is None:
        return {"verdict": "CORPUS_UNAVAILABLE", "matches": []}
    with mp.workdps(digits + 20):
        value = mp.mpf(value_text)
        window = mp.mpf(10) ** (-digits)
        matches = []
        for row in corpus_rows:
            try:
                other = mp.mpf(row["value"])
            except (ValueError, TypeError):
                continue
            scale = max(mp.mpf(1), abs(other))
            if abs(value - other) / scale < window:
                matches.append(
                    {
                        "record_id": row["record_id"],
                        "family": row["family"],
                        "value_expr": row["value_expr"],
                        "citation": row["citation"],
                    }
                )
                if len(matches) >= 4:
                    break
    if matches:
        return {"verdict": "CONSTANT_ATTESTED_IN_CORPUS", "matches": matches}
    return {"verdict": "NOT_FOUND_IN_CORPUS", "matches": []}


def classical_reduction(
    tokens: Sequence[int], submode: str, target_name: str
) -> dict[str, Any]:
    """Try the declared classical reductions in order, recording every one that declined.

    ``constant`` programs are attacked symbolically (a mechanised elementary identity is a
    proof, and it makes the candidate known) and then by PSLQ against the classical basis.
    ``series`` programs go to :mod:`sigma_theory_compiler.cf_proof_router`'s hypergeometric
    machinery: if the term ratio is a rational function of ``k``, the sum is a ``pFq`` and the
    router evaluates it in closed form.  ``limit`` programs are attacked symbolically.
    """

    attempts: list[dict[str, Any]] = []
    expression = to_sympy(tokens)
    if expression is None:
        return {
            "verdict": "NOT_REDUCED",
            "techniques_attempted": [{"technique": "parse", "blocker": "invalid_program"}],
        }
    target = SYMBOLIC_TARGETS.get(target_name)

    if submode in {"constant", "limit"}:
        subject = expression
        if submode == "limit":
            try:
                subject = sp.limit(expression, _SYMPY_K, sp.oo)
            except (NotImplementedError, ValueError, TypeError, RecursionError) as error:
                attempts.append(
                    {"technique": "symbolic_limit", "fired": False, "blocker": str(error)[:120]}
                )
                subject = None
        if subject is not None and target is not None:
            try:
                residual = sp.simplify(subject - target)
            except (NotImplementedError, ValueError, TypeError, RecursionError) as error:
                residual = None
                attempts.append(
                    {
                        "technique": "symbolic_identity",
                        "fired": False,
                        "blocker": f"sympy declined: {str(error)[:120]}",
                    }
                )
            if residual is not None:
                if residual == 0:
                    return {
                        "verdict": "KNOWN_BY_PROOF_FAMILY",
                        "technique_that_fired": "symbolic_identity",
                        "cited_theorem": (
                            "mechanised elementary-function identity proved by sympy over the "
                            "rationals, radicals, exp/log and inverse trigonometric functions"
                        ),
                        "derivation": f"simplify({sp.srepr(subject)[:200]} - target) = 0",
                        "techniques_attempted": attempts,
                    }
                attempts.append(
                    {
                        "technique": "symbolic_identity",
                        "fired": False,
                        "blocker": "residual did not simplify to zero",
                    }
                )
        outcome = _pslq_reduction(subject if subject is not None else expression, target_name)
        attempts.append(outcome["summary"])
        if outcome["fired"]:
            return {
                "verdict": "KNOWN_BY_PROOF_FAMILY",
                "technique_that_fired": "pslq_rational_linear_in_classical_basis",
                "cited_theorem": (
                    "integer-relation detection (PSLQ, Ferguson-Bailey-Arno 1999) against the "
                    "declared classical constant basis"
                ),
                "derivation": outcome["derivation"],
                "techniques_attempted": attempts,
            }
        return {"verdict": "NOT_REDUCED", "techniques_attempted": attempts}

    from .cf_proof_router import (
        evaluate_hypergeometric,
        hypergeometric_series_value,
        pfq_parameters,
    )

    symbol = sp.Symbol("k")
    try:
        ratio = sp.cancel(sp.simplify(expression.subs(_SYMPY_K, _SYMPY_K + 1) / expression))
        ratio = ratio.subs(_SYMPY_K, symbol)
    except (NotImplementedError, ValueError, TypeError, RecursionError) as error:
        attempts.append({"technique": "term_ratio", "fired": False, "blocker": str(error)[:120]})
        ratio = None
    if ratio is None or ratio.free_symbols - {symbol}:
        attempts.append(
            {
                "technique": "hypergeometric_term_test",
                "fired": False,
                "blocker": "term ratio is not a function of k alone",
            }
        )
        return {"verdict": "NOT_REDUCED", "techniques_attempted": attempts}
    if not sp.together(ratio).is_rational_function(symbol):
        attempts.append(
            {
                "technique": "hypergeometric_term_test",
                "fired": False,
                "blocker": (
                    "term ratio is not a rational function of k: the term is not "
                    "hypergeometric, so no pFq route applies"
                ),
            }
        )
        outcome = _pslq_reduction(_series_value_expression(tokens), target_name)
        attempts.append(outcome["summary"])
        if outcome["fired"]:
            return _pslq_verdict(outcome, attempts)
        return {"verdict": "NOT_REDUCED", "techniques_attempted": attempts}

    first = expression.subs(_SYMPY_K, SERIES_START_INDEX)
    # Route 1: the router's own path, guard and all.
    try:
        value, reduction, detail = hypergeometric_series_value(ratio, SERIES_START_INDEX)
    except (NotImplementedError, ValueError, TypeError, RecursionError) as error:
        value, reduction, detail = None, f"router raised: {str(error)[:120]}", {}
    if value is not None:
        return _hypergeometric_verdict(
            "hypergeometric_closed_form_via_cf_proof_router",
            ratio,
            detail,
            reduction,
            sp.simplify(first * value),
            attempts,
        )
    attempts.append(
        {
            "technique": "hypergeometric_closed_form_via_cf_proof_router",
            "fired": False,
            "blocker": reduction,
        }
    )

    # Route 2: the router's guard sums the series directly, which needs geometric-ish decay.
    # An algebraically converging series (1/k^2 and friends) never reaches its tolerance in
    # budget, so the closed form is re-guarded here against an *accelerated* sum of the same
    # series.  The hypergeometric machinery is still the router's; only the guard changes.
    parameters = pfq_parameters(sp.cancel(ratio.subs(symbol, sp.Symbol("m") + SERIES_START_INDEX)))
    if parameters is not None:
        a_list, b_list, argument = parameters
        rendered = (
            f"{len(a_list)}F{len(b_list)}("
            + ", ".join(str(item) for item in a_list)
            + "; "
            + ", ".join(str(item) for item in b_list)
            + f"; {argument})"
        )
        closed, how = evaluate_hypergeometric(a_list, b_list, argument)
        if closed is not None:
            total = sp.simplify(first * closed)
            with mp.workdps(60):
                summed = evaluate_series_mp(tokens)
                if summed is not None:
                    try:
                        predicted = mp.mpf(str(sp.N(total, 50)))
                    except (TypeError, ValueError):
                        predicted = None
                    if predicted is not None and abs(predicted - summed) < mp.mpf("1e-30"):
                        return _hypergeometric_verdict(
                            "hypergeometric_closed_form_via_cf_proof_router",
                            ratio,
                            {
                                "series": rendered,
                                "numeric_guard": (
                                    "closed form agrees with the accelerated (mpmath.nsum) sum "
                                    "of the same series to 30 digits at 60-digit precision"
                                ),
                            },
                            f"{how}+accelerated_sum_guard",
                            total,
                            attempts,
                        )
            attempts.append(
                {
                    "technique": "hypergeometric_closed_form_with_accelerated_guard",
                    "fired": False,
                    "blocker": "closed form disagreed with the accelerated sum",
                }
            )
        # Route 3: no closed form, but the term ratio is rational in k, so the sum *is* a
        # generalised hypergeometric series.  Membership in the pFq family is itself classical
        # prior art (Gauss 1812); claiming novelty for a pFq would be wrong even when sympy
        # cannot expand it.
        return {
            "verdict": "KNOWN_BY_PROOF_FAMILY",
            "technique_that_fired": "hypergeometric_term_pfq_identification",
            "cited_theorem": (
                "Gauss 1812: a series whose term ratio is a rational function of the index is a "
                "generalised hypergeometric series; the pFq is exhibited explicitly"
            ),
            "derivation": {
                "term_ratio": str(ratio),
                "series": rendered,
                "reduction": "pfq_parameters_read_off_the_term_ratio",
                "closed_form": None,
                "closed_form_blocker": how,
            },
            "techniques_attempted": attempts,
        }
    attempts.append(
        {
            "technique": "hypergeometric_term_pfq_identification",
            "fired": False,
            "blocker": "term ratio did not split into rational linear factors",
        }
    )
    outcome = _pslq_reduction(_series_value_expression(tokens), target_name)
    attempts.append(outcome["summary"])
    if outcome["fired"]:
        return _pslq_verdict(outcome, attempts)
    return {"verdict": "NOT_REDUCED", "techniques_attempted": attempts}


def _hypergeometric_verdict(
    technique: str,
    ratio: sp.Expr,
    detail: Mapping[str, Any],
    reduction: str,
    closed_form: sp.Expr,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "verdict": "KNOWN_BY_PROOF_FAMILY",
        "technique_that_fired": technique,
        "cited_theorem": (
            "Gauss/Euler hypergeometric summation as mechanised in "
            "cf_proof_router.hypergeometric_series_value"
        ),
        "derivation": {
            "term_ratio": str(ratio),
            "series": detail.get("series"),
            "reduction": reduction,
            "closed_form": str(closed_form),
            "numeric_guard": detail.get("numeric_guard"),
        },
        "techniques_attempted": attempts,
    }


def _pslq_verdict(outcome: Mapping[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "verdict": "KNOWN_BY_PROOF_FAMILY",
        "technique_that_fired": "pslq_rational_linear_in_classical_basis",
        "cited_theorem": (
            "integer-relation detection (PSLQ, Ferguson-Bailey-Arno 1999) against the declared "
            "classical constant basis"
        ),
        "derivation": outcome["derivation"],
        "techniques_attempted": attempts,
    }


def _series_value_expression(tokens: Sequence[int]) -> mp.mpf | None:
    """The series' own value at PSLQ precision, so the integer-relation lane can see it."""

    with mp.workdps(PSLQ_DPS + 20):
        return evaluate_series_mp(tokens)


PSLQ_BASIS: tuple[str, ...] = (
    "one",
    "pi",
    "e",
    "ln2",
    "ln3",
    "sqrt2",
    "sqrt3",
    "zeta3",
    "euler_gamma",
    "catalan",
)
PSLQ_DPS = 120
PSLQ_MAX_COEFFICIENT = 10**6
PSLQ_MAX_STEPS = 10000


def _basis_value(name: str) -> mp.mpf:
    if name == "one":
        return mp.mpf(1)
    return target_value_mp(name)


#: Memo for the integer-relation lane.  Hundreds of survivors share one constant -- every
#: padded variant of ``4 arctan 1`` is the same real number -- and PSLQ is a pure function of
#: that number, so the cache changes the runtime and nothing else.
_PSLQ_CACHE: dict[str, dict[str, Any]] = {}


def _pslq_reduction(subject: sp.Expr | mp.mpf | None, target_name: str) -> dict[str, Any]:
    """Integer relation between the candidate's constant and the declared classical basis."""

    technique = "pslq_rational_linear_in_classical_basis"
    if subject is None:
        return {
            "fired": False,
            "summary": {
                "technique": technique,
                "fired": False,
                "blocker": "no numeric value available for the candidate",
            },
        }
    with mp.workdps(PSLQ_DPS):
        try:
            value = (
                mp.mpf(subject)
                if isinstance(subject, mp.mpf)
                else mp.mpf(str(sp.N(subject, PSLQ_DPS)))
            )
        except (TypeError, ValueError, NotImplementedError):
            return {
                "fired": False,
                "summary": {
                    "technique": technique,
                    "fired": False,
                    "blocker": "candidate did not evaluate numerically",
                },
            }
        cached = _PSLQ_CACHE.get(mp.nstr(value, 60))
        if cached is not None:
            return cached
        vector = [value] + [_basis_value(name) for name in PSLQ_BASIS]
        try:
            relation = mp.pslq(
                vector,
                tol=mp.mpf("1e-100"),
                maxcoeff=PSLQ_MAX_COEFFICIENT,
                maxsteps=PSLQ_MAX_STEPS,
            )
        except (ValueError, ZeroDivisionError, ArithmeticError):
            relation = None
    with mp.workdps(PSLQ_DPS):
        cache_key = mp.nstr(value, 60)
    if relation is None or relation[0] == 0:
        outcome: dict[str, Any] = {
            "fired": False,
            "summary": {
                "technique": technique,
                "fired": False,
                "blocker": (
                    "no integer relation with coefficients under "
                    f"{PSLQ_MAX_COEFFICIENT} at {PSLQ_DPS} digits (a bounded-search fact, not "
                    "a proof that none exists)"
                ),
            },
        }
        _PSLQ_CACHE[cache_key] = outcome
        return outcome
    coefficients = [int(item) for item in relation]
    terms = {
        name: coefficient
        for name, coefficient in zip(PSLQ_BASIS, coefficients[1:], strict=True)
        if coefficient
    }
    outcome = {
        "fired": True,
        "derivation": {
            "coefficient_on_candidate": coefficients[0],
            "coefficients_on_basis": terms,
            "basis": list(PSLQ_BASIS),
            "fit_dps": PSLQ_DPS,
        },
        "summary": {"technique": technique, "fired": True},
    }
    _PSLQ_CACHE[cache_key] = outcome
    return outcome


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def reachability_controls() -> list[dict[str, Any]]:
    """Every built-in classical identity must live in the declared space and evaluate right.

    This is the control that decides whether the enumerator reaches real mathematics at all.
    A generative grammar that cannot express ``4 arctan 1``, Euler's Machin-like formula, the
    Basel series, Apery's series, the golden ratio through a square root, and Bernoulli's
    limit for ``e`` is not searching mathematics; it is searching noise.
    """

    rows: list[dict[str, Any]] = []
    for item in BUILTIN_KNOWN_IDENTITIES:
        tokens = _tokens_from_rpn(str(item["rpn"]))
        mode = str(item["mode"])
        length = int(MODE_CONFIG[mode]["program_length"])
        status = program_status(tokens)
        ordinal = None
        roundtrip = False
        if len(tokens) == length and status == "ok":
            ordinal = encode_program(tokens, mode)
            roundtrip = decode_ordinal(ordinal, mode) == tokens
        with mp.workdps(50):
            value = (
                evaluate_program_value_mp(tokens, str(item["submode"]))
                if roundtrip
                else None
            )
            target = target_value_mp(str(item["target"]))
            if value is None:
                distance = None
                agrees = False
            else:
                scaled = abs(value - target) / max(mp.mpf(1), abs(target))
                distance = mp.nstr(scaled, 4)
                agrees = bool(scaled < mp.mpf(10) ** -40)
        rows.append(
            {
                "id": item["id"],
                "attribution": item["attribution"],
                "mode": mode,
                "submode": item["submode"],
                "target": item["target"],
                "rpn": item["rpn"],
                "infix": render_infix(tokens),
                "program_length": len(tokens),
                "structural_status": status,
                "ordinal": ordinal,
                "codec_roundtrip": roundtrip,
                "scaled_distance_at_50dps": distance,
                "reproduces_target": agrees,
            }
        )
    return rows


def _forge_reachability_upgrade(candidate: dict[str, Any]) -> None:
    """Relabel one unresolved reachability cell as a publishable negative, in place.

    This is the forgery the C1 gate exists to stop, expressed at the receipt level: the sweep
    found nothing for a target whose reachability was never established, and the receipt claims
    that silence as a result anyway.
    """

    from .certified_null_search import OUTCOME_REAL_NEGATIVE
    from .reachability_certificate import VERDICT_UNRESOLVED, seal_certificate

    block = candidate.get("reachability")
    if not block:  # pragma: no cover - defensive
        raise CompositionalSearchError("tamper probe needs a reachability block to forge")
    row = next(
        item
        for item in block["null_adjudication"]
        if item["certificate_verdict"] == VERDICT_UNRESOLVED
    )
    row["outcome"] = OUTCOME_REAL_NEGATIVE
    row["publishable_as_a_negative"] = True
    row["uninformative_reason"] = None
    candidate["reachability"] = seal_certificate(block)


def tamper_control() -> dict[str, Any]:
    """The seal must reject a silently edited receipt and a re-sealed dishonest one.

    Run in-campaign rather than only in the tests, because a receipt whose seal does not bite
    is worth nothing and the run should not finish pretending otherwise.

    The synthetic receipt reports a null, so it also exercises the C1 gate on every run: it
    carries a real reachability block, and two of the probes below try to publish that null
    without one -- by stripping the block, and by upgrading an unresolved cell to a real
    negative.  Both must be refused, or the gate is decoration.
    """

    from .certified_null_search import campaign_reachability_block

    reachability = campaign_reachability_block({"C": {}, "F": {}})
    body = {
        "schema_version": RESULT_SCHEMA,
        "claims": dict(CLAIMS),
        "decision": "SEARCHED",
        "codec_validity": [{"mode": "C", "agrees_with_closed_form": True}],
        "controls": {
            "reachability": [{"id": "tamper_probe", "reproduces_target": True}],
            "determinism": [
                {"identical_accumulators": True, "identical_retained_ordinals": True}
            ],
            "cpu_gpu_crosscheck": [{"passes": True}],
        },
        "decoy_calibration": {"totals": {"decoy_post_gate_survivors": 0}},
        "reachability": reachability,
        "headline": {"count": 0, "entries": []},
    }
    body["result_core_sha256"] = canonical_sha256(body)
    body["measurement"] = {"elapsed_seconds": "0.000"}
    sealed = {**body, "content_sha256": canonical_sha256(body)}

    probes: list[dict[str, Any]] = []

    def probe(name: str, mutate: Any) -> None:
        candidate = json.loads(json.dumps(sealed))
        mutate(candidate)
        try:
            validate_receipt(candidate)
        except CompositionalSearchError as error:
            probes.append({"probe": name, "rejected": True, "reason": str(error)})
            return
        probes.append({"probe": name, "rejected": False, "reason": None})

    def reseal(candidate: dict[str, Any]) -> None:
        core = {
            key: value
            for key, value in candidate.items()
            if key not in {"content_sha256", "result_core_sha256", "measurement"}
        }
        candidate["result_core_sha256"] = canonical_sha256(core)
        candidate["content_sha256"] = canonical_sha256(
            {key: value for key, value in candidate.items() if key != "content_sha256"}
        )

    probe("silent_edit_without_resealing", lambda c: c.update({"decision": "TAMPERED"}))
    probe(
        "flipped_claim_with_a_fresh_seal",
        lambda c: (
            c["claims"].update({"corpus_absence_establishes_novelty": True}),
            reseal(c),
        ),
    )
    probe(
        "surviving_decoy_with_a_fresh_seal",
        lambda c: (
            c["decoy_calibration"]["totals"].update({"decoy_post_gate_survivors": 1}),
            reseal(c),
        ),
    )
    probe(
        "reduced_candidate_promoted_to_the_headline",
        lambda c: (
            c.update(
                {
                    "headline": {
                        "count": 1,
                        "entries": [
                            {
                                "survived_chance_gate": True,
                                "classical_reduction": {
                                    "verdict": "KNOWN_BY_PROOF_FAMILY"
                                },
                            }
                        ],
                    }
                }
            ),
            reseal(c),
        ),
    )
    probe(
        "null_published_after_stripping_the_reachability_block",
        lambda c: (c.pop("reachability", None), reseal(c)),
    )
    probe(
        "unresolved_reachability_upgraded_to_a_real_negative",
        lambda c: (_forge_reachability_upgrade(c), reseal(c)),
    )
    honest = True
    try:
        validate_receipt(sealed)
    except CompositionalSearchError:
        honest = False
    return {
        "untampered_receipt_validates": honest,
        "probes": probes,
        "all_probes_rejected": all(item["rejected"] for item in probes),
    }


def determinism_control(engine: _Engine, start: int, count: int) -> dict[str, Any]:
    """The same ordinal range twice: every accumulator must come back bit-identical."""

    parameters = SweepParameters(engine.n_sub)
    for index in range(engine.n_sub * SLOT_COUNT):
        parameters.window_core[index] = MODE_F_NEAR_WINDOW if engine.mode == "F" else MATCH_WINDOW
        parameters.threshold_core[index] = (1 << 64) - 1
    digests: list[str] = []
    hits: list[tuple[int, ...]] = []
    for _ in range(2):
        engine.reset()
        found = engine.launch(start, count, parameters)
        digests.append(
            canonical_sha256(
                {
                    "window": _encode_array(engine.window_counts.get()),
                    "status": _encode_array(engine.status_counts.get()),
                    "mag": _encode_array(engine.mag_hist.get()),
                    "hll": _encode_array(engine.hll.get()),
                }
            )
        )
        hits.append(tuple(int(value) for value in found[0]))
    engine.reset()
    return {
        "mode": engine.mode,
        "start": start,
        "count": count,
        "accumulator_sha256": digests,
        "identical_accumulators": digests[0] == digests[1],
        "identical_retained_ordinals": hits[0] == hits[1],
        "retained": len(hits[0]),
    }


def crosscheck_control(engine: _Engine, total: int, sample: int = CROSSCHECK_SAMPLE) -> dict[str, Any]:
    """GPU against the numpy reference machine on a deterministic ordinal sample."""

    rng = np.random.default_rng(int(hashlib.sha256(DECOY_SEED.encode()).hexdigest()[:8], 16))
    ordinals = np.sort(rng.choice(total, size=min(sample, total), replace=False)).astype(np.int64)
    gpu_values, gpu_flags = engine.evaluate_list(ordinals)
    if engine.mode == "C":
        cpu_values, cpu_usable = evaluate_batch_cpu(ordinals, "C")
        gpu_usable = gpu_flags.astype(bool)
        agree_flags = int(np.count_nonzero(gpu_usable == cpu_usable))
        both = gpu_usable & cpu_usable
        difference = np.abs(gpu_values[both] - cpu_values[both]) / np.maximum(
            1.0, np.abs(cpu_values[both])
        )
        worst = float(difference.max()) if difference.size else 0.0
        checked = int(both.sum())
    else:
        cpu_series, cpu_limit, cpu_flags = evaluate_series_batch_cpu(
            ordinals, "F", STAGE1_TERMS, STAGE1_LADDER
        )
        gpu_series = gpu_values[:, 0]
        gpu_limit = gpu_values[:, 1]
        gpu_series_ok = (gpu_flags & 1).astype(bool)
        gpu_limit_ok = (gpu_flags & 2).astype(bool)
        agree_flags = int(
            np.count_nonzero(gpu_series_ok == cpu_flags[:, 0])
            + np.count_nonzero(gpu_limit_ok == cpu_flags[:, 1])
        )
        both_series = gpu_series_ok & cpu_flags[:, 0]
        both_limit = gpu_limit_ok & cpu_flags[:, 1]
        parts = []
        if both_series.any():
            parts.append(
                np.abs(gpu_series[both_series] - cpu_series[both_series])
                / np.maximum(1.0, np.abs(cpu_series[both_series]))
            )
        if both_limit.any():
            parts.append(
                np.abs(gpu_limit[both_limit] - cpu_limit[both_limit])
                / np.maximum(1.0, np.abs(cpu_limit[both_limit]))
            )
        difference = np.concatenate(parts) if parts else np.empty(0)
        worst = float(difference.max()) if difference.size else 0.0
        checked = int(both_series.sum() + both_limit.sum())
        agree_flags = agree_flags // 2
    return {
        "mode": engine.mode,
        "sample": int(ordinals.size),
        "flag_agreements": agree_flags,
        "flag_disagreements": int(ordinals.size) - agree_flags,
        "values_compared": checked,
        "worst_scaled_difference": format(worst, ".3e"),
        "tolerance": "1e-12",
        "passes": bool(agree_flags == int(ordinals.size) and worst < 1e-12),
    }


#: A dedup cross-check over a sample holding fewer usable values than this is vacuous, and a
#: vacuous control must not report itself green.
DEDUP_MIN_USABLE = 1000


def dedup_exact_control(engine: _Engine, total: int, sample: int) -> dict[str, Any]:
    """HyperLogLog against an exact distinct count on a declared ordinal sub-sample.

    The sample is drawn across the whole space from the declared seed rather than taken as a
    contiguous block: valid programs are wildly unevenly distributed over the ordinal range --
    the top of Mode C's space is all ``pow`` tokens and holds no valid program at all -- so a
    block sample can come back empty and call itself a pass.
    """

    rng = np.random.default_rng(
        int(hashlib.sha256(f"{DECOY_SEED}|dedup".encode()).hexdigest()[:8], 16)
    )
    ordinals = np.sort(rng.choice(total, size=min(sample, total), replace=False)).astype(np.int64)
    values, flags = engine.evaluate_list(ordinals)
    if engine.mode == "C":
        usable = flags.astype(bool)
        selected = values[usable]
    else:
        usable = (flags & 1).astype(bool)
        selected = values[:, 0][usable]
    keys = value_hash_keys(selected)
    exact = int(np.unique(keys).size)
    hashed = np.array([_splitmix64(int(key)) for key in keys], dtype=np.uint64)
    registers = hll_registers_from_hashes(hashed)
    estimate = hll_estimate(registers)
    error = abs(estimate - exact) / max(1.0, exact)
    return {
        "mode": engine.mode,
        "sample_ordinals": int(sample),
        "usable_values": int(selected.size),
        "exact_distinct": exact,
        "hyperloglog_estimate": format(estimate, ".6e"),
        "relative_error": format(error, ".4f"),
        "declared_standard_error": format(HLL_STANDARD_ERROR, ".4f"),
        "minimum_usable_for_a_meaningful_check": DEDUP_MIN_USABLE,
        "passes": bool(
            selected.size >= DEDUP_MIN_USABLE and error <= 6 * HLL_STANDARD_ERROR + 0.02
        ),
    }


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


def _magnitude_summary(histogram: np.ndarray) -> dict[str, Any]:
    total = int(histogram.sum())
    if total == 0:
        return {"values": 0, "buckets": []}
    cumulative = np.cumsum(histogram)
    quantiles: dict[str, str] = {}
    for label, fraction in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95)):
        index = int(np.searchsorted(cumulative, fraction * total))
        index = min(index, NMAG - 1)
        quantiles[label] = f"2^{index * 8 - 352}..2^{index * 8 - 345}"
    top = np.argsort(histogram)[::-1][:8]
    return {
        "values": total,
        "log2_magnitude_quantiles": quantiles,
        "busiest_buckets": [
            {
                "log2_range": f"[{int(index) * 8 - 352}, {int(index) * 8 - 344})",
                "count": int(histogram[index]),
                "fraction": format(float(histogram[index]) / total, ".4f"),
            }
            for index in top
            if histogram[index]
        ],
    }


def _submode_hits(
    hits: tuple[np.ndarray, np.ndarray, np.ndarray], submodes: Sequence[str]
) -> dict[tuple[str, int, int], np.ndarray]:
    """Group retained hits by (submode, slot, tier), each sorted by ordinal."""

    ordinals, tags, values = hits
    grouped: dict[tuple[str, int, int], np.ndarray] = {}
    if ordinals.size == 0:
        return grouped
    scales = np.maximum(1.0, np.abs(SLOT_VALUES))
    tier = tags % 2
    index = tags // 2
    submode_index = index // SLOT_COUNT
    slot_index = index % SLOT_COUNT
    for sub_position, submode in enumerate(submodes):
        for slot in range(SLOT_COUNT):
            for tier_value in (0, 1):
                mask = (
                    (submode_index == sub_position)
                    & (slot_index == slot)
                    & (tier == tier_value)
                )
                if not mask.any():
                    continue
                distance = np.abs(values - SLOT_VALUES[slot]) / scales[slot]
                # The stress tier is the annulus, never the atom.
                mask = mask & (
                    (distance < CORE_WINDOW)
                    if tier_value == 0
                    else ((distance >= CORE_WINDOW) & (distance < STRESS_WINDOW))
                )
                if not mask.any():
                    continue
                grouped[(submode, slot, tier_value)] = np.stack(
                    [ordinals[mask].astype(np.float64), values[mask]], axis=1
                )
    return grouped


def _verify_group(
    entries: np.ndarray,
    mode: str,
    submode: str,
    slot: Mapping[str, Any],
    required_digits: int,
    dps: int,
    budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify the closest ``budget`` candidates of one (submode, slot, tier) group."""

    target = float(slot["value"])
    scale = max(1.0, abs(target))
    distance = np.abs(entries[:, 1] - target) / scale
    order = np.lexsort((entries[:, 0], distance))
    chosen = order[:budget]
    reports = [
        verify_candidate(int(entries[index, 0]), mode, submode, slot, required_digits, dps)
        for index in chosen
    ]
    return reports, {
        "retained": int(entries.shape[0]),
        "verified": int(chosen.size),
        "verification_complete": bool(chosen.size == entries.shape[0]),
        "selection_rule": "closest fp64 scaled distance first, ordinal as tie-break",
    }


def _mode_f_fine_counts(
    census_counts: np.ndarray,
    stage2: Mapping[str, Any],
    submodes: Sequence[str],
) -> np.ndarray:
    """Replace Mode F's fine window counts with exact ones taken from the stage-2 values."""

    counts = census_counts.copy()
    floor_index = WINDOW_EXPONENTS.index(STAGE1_WINDOW_FLOOR_EXPONENT)
    for sub_index, submode in enumerate(submodes):
        values = stage2["values"][submode]
        if values.size == 0:
            counts[sub_index, :, floor_index + 1 :] = 0
            continue
        for slot in TARGET_SLOTS:
            slot_index = int(slot["slot"])
            target = float(slot["value"])
            scale = max(1.0, abs(target))
            distance = np.abs(values - target) / scale
            for window_index in range(floor_index + 1, len(WINDOWS)):
                counts[sub_index, slot_index, window_index] = int(
                    np.count_nonzero(distance < WINDOWS[window_index])
                )
    return counts


def run_campaign(
    *,
    root: Path,
    mode_c_limit: int | None = None,
    mode_f_limit: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    launch_size: int = DEFAULT_LAUNCH_SIZE,
    checkpoint_dir: Path | None = None,
    verify_budget_c: int = VERIFY_BUDGET_PER_SLOT,
    verify_budget_f: int = 32,
    skip_corpus: bool = False,
) -> dict[str, Any]:
    """Run the full compositional search and seal a receipt."""

    started = time.perf_counter()
    engines: dict[str, _Engine] = {}
    totals: dict[str, int] = {}
    for mode, limit in (("C", mode_c_limit), ("F", mode_f_limit)):
        engines[mode] = _Engine(mode)
        space = int(MODE_CONFIG[mode]["space_size"])
        totals[mode] = space if limit is None else min(int(limit), space)
    device = engines["C"].device

    controls: dict[str, Any] = {"reachability": reachability_controls()}
    if not all(row["reproduces_target"] for row in controls["reachability"]):
        failed = [row["id"] for row in controls["reachability"] if not row["reproduces_target"]]
        raise CompositionalSearchError(
            f"rediscovery control failed: the grammar did not reproduce {failed}"
        )

    # The enumerator's own evaluator, applied to the control ordinals it does visit.
    # Each stage is held to the window it is responsible for.  Mode F stage 1 only has to put
    # the control inside the near-target set that stage 2 then re-sums; stage 2 has to land it
    # inside the fp64 match window.  Holding stage 1 to the match window would be holding a
    # 64-term partial sum to a standard it structurally cannot meet.
    control_gpu: list[dict[str, Any]] = []
    for row in controls["reachability"]:
        mode = str(row["mode"])
        ordinal = int(row["ordinal"])
        with mp.workdps(30):
            target = float(target_value_mp(str(row["target"])))
        scale = max(1.0, abs(target))
        entry: dict[str, Any] = {
            "id": row["id"],
            "ordinal": ordinal,
            "inside_enumerated_range": ordinal < totals[mode],
        }
        values, flags = engines[mode].evaluate_list(np.array([ordinal], dtype=np.int64))
        if mode == "C":
            usable = bool(flags[0] & 1)
            distance = abs(float(values[0]) - target) / scale
            entry.update(
                {
                    "gpu_usable": usable,
                    "gpu_scaled_distance": format(distance, ".3e"),
                    "window_required": format(MATCH_WINDOW, ".0e"),
                    "gpu_matches_in_required_window": bool(usable and distance < MATCH_WINDOW),
                }
            )
        else:
            index = 0 if row["submode"] == "series" else 1
            usable = bool(int(flags[0]) & (1 << index))
            distance = abs(float(values[0, index]) - target) / scale
            series, limit, flags2 = engines[mode].stage2(np.array([ordinal], dtype=np.int64))
            usable2 = bool(int(flags2[0]) & (1 << index))
            value2 = float(series[0] if index == 0 else limit[0])
            distance2 = abs(value2 - target) / scale
            entry.update(
                {
                    "gpu_usable": usable,
                    "gpu_scaled_distance": format(distance, ".3e"),
                    "window_required": format(MODE_F_NEAR_WINDOW, ".0e"),
                    "stage1_reaches_near_target_set": bool(
                        usable and distance < MODE_F_NEAR_WINDOW
                    ),
                    "stage2_usable": usable2,
                    "stage2_scaled_distance": format(distance2, ".3e"),
                    "gpu_matches_in_required_window": bool(
                        usable
                        and distance < MODE_F_NEAR_WINDOW
                        and usable2
                        and distance2 < MATCH_WINDOW
                    ),
                }
            )
        control_gpu.append(entry)
    controls["reachability_on_gpu"] = control_gpu
    missed = [
        row["id"]
        for row in control_gpu
        if row["inside_enumerated_range"] and not row["gpu_matches_in_required_window"]
    ]
    if missed:
        raise CompositionalSearchError(
            f"rediscovery control failed on the GPU evaluator for {missed}"
        )

    controls["receipt_tamper"] = tamper_control()
    if not (
        controls["receipt_tamper"]["untampered_receipt_validates"]
        and controls["receipt_tamper"]["all_probes_rejected"]
    ):
        raise CompositionalSearchError("receipt tamper control failed: the seal does not bite")

    controls["determinism"] = [
        determinism_control(engines["C"], int(MODE_CONFIG["C"]["space_size"]) - (1 << 22), 1 << 22),
        determinism_control(engines["F"], int(MODE_CONFIG["F"]["space_size"]) - (1 << 20), 1 << 20),
    ]
    if not all(
        row["identical_accumulators"] and row["identical_retained_ordinals"]
        for row in controls["determinism"]
    ):
        raise CompositionalSearchError("determinism control failed: a repeat sweep differed")

    controls["cpu_gpu_crosscheck"] = [
        crosscheck_control(engines["C"], totals["C"]),
        crosscheck_control(engines["F"], totals["F"], sample=1 << 16),
    ]
    if not all(row["passes"] for row in controls["cpu_gpu_crosscheck"]):
        raise CompositionalSearchError("CPU/GPU cross-check disagreed")

    # --- the sweeps -------------------------------------------------------
    sweeps: dict[str, dict[str, Any]] = {}
    for mode in ("C", "F"):
        engine = engines[mode]
        submodes = engine.submodes
        engine.reset()
        empty = SweepParameters(engine.n_sub)
        census = run_sweep(
            engine,
            pass_name="census",
            total=totals[mode],
            parameters=empty,
            chunk_size=chunk_size,
            launch_size=launch_size,
            checkpoint_path=(
                None if checkpoint_dir is None else checkpoint_dir / f"{mode}-census.json"
            ),
            hit_budget=0,
        )
        retention_parameters = derive_retention(census, submodes, mode)
        engine.reset()
        retention = run_sweep(
            engine,
            pass_name="retention",
            total=totals[mode],
            parameters=retention_parameters,
            chunk_size=chunk_size,
            launch_size=launch_size,
            checkpoint_path=(
                None if checkpoint_dir is None else checkpoint_dir / f"{mode}-retention.json"
            ),
            hit_budget=(1 << 20) if mode == "C" else RETENTION_BUDGET_MODE_F_NEAR,
        )
        sweeps[mode] = {
            "census": census,
            "retention": retention,
            "retention_parameters": retention_parameters,
        }

    # --- Mode F stage 2: the near-target set re-summed at 64x the depth ----
    engine_f = engines["F"]
    retained_f = sweeps["F"]["retention"]["hits"]
    unique_f = np.unique(retained_f[0]) if retained_f[0].size else np.empty(0, dtype=np.int64)
    stage2_series = np.empty(0, dtype=np.float64)
    stage2_limit = np.empty(0, dtype=np.float64)
    stage2_flags = np.empty(0, dtype=np.uint8)
    for start in range(0, unique_f.size, 1 << 20):
        block = unique_f[start : start + (1 << 20)]
        series, limit, flags = engine_f.stage2(block)
        stage2_series = np.concatenate([stage2_series, series])
        stage2_limit = np.concatenate([stage2_limit, limit])
        stage2_flags = np.concatenate([stage2_flags, flags])
    stage2 = {
        "ordinals": unique_f,
        "values": {
            "series": stage2_series[(stage2_flags & 1).astype(bool)],
            "limit": stage2_limit[(stage2_flags & 2).astype(bool)],
        },
        "series_ordinals": unique_f[(stage2_flags & 1).astype(bool)],
        "limit_ordinals": unique_f[(stage2_flags & 2).astype(bool)],
        "series_values": stage2_series,
        "limit_values": stage2_limit,
        "flags": stage2_flags,
        "terms": STAGE2_TERMS,
        "ladder": STAGE2_LADDER,
    }

    mode_f_complete = not bool(sweeps["F"]["retention"]["hits_truncated"])
    counts = {
        "C": sweeps["C"]["census"]["window_counts"],
        "F": _mode_f_fine_counts(sweeps["F"]["census"]["window_counts"], stage2, engine_f.submodes),
    }
    if not mode_f_complete:
        raise CompositionalSearchError(
            "Mode F near-target retention hit its budget, so the fine window counts would be "
            "lower bounds rather than measurements; raise RETENTION_BUDGET_MODE_F_NEAR and "
            "re-run rather than publishing a count that is not a count"
        )
    gate = {
        mode: chance_accounting(counts[mode], tuple(MODE_CONFIG[mode]["submodes"]))
        for mode in ("C", "F")
    }
    verification_dps = max(gate["C"]["verification_dps"], gate["F"]["verification_dps"])
    required_by_key = {
        (mode, row["submode"], row["target"]): int(row["required_agreement_digits"])
        for mode in ("C", "F")
        for row in gate[mode]["per_target"]
    }

    # --- verification at the derived precision ----------------------------
    verified: list[dict[str, Any]] = []
    verification_summary: list[dict[str, Any]] = []
    for mode in ("C", "F"):
        submodes = tuple(MODE_CONFIG[mode]["submodes"])
        budget = verify_budget_c if mode == "C" else verify_budget_f
        if mode == "C":
            groups = _submode_hits(sweeps["C"]["retention"]["hits"], submodes)
        else:
            groups = {}
            for sub_position, submode in enumerate(submodes):
                ordinals = (
                    stage2["series_ordinals"] if submode == "series" else stage2["limit_ordinals"]
                )
                values = stage2["values"][submode]
                if ordinals.size == 0:
                    continue
                for slot in TARGET_SLOTS:
                    target = float(slot["value"])
                    scale = max(1.0, abs(target))
                    distance = np.abs(values - target) / scale
                    for tier in (0, 1):
                        mask = (
                            distance < CORE_WINDOW
                            if tier == 0
                            else ((distance >= CORE_WINDOW) & (distance < STRESS_WINDOW))
                        )
                        if not mask.any():
                            continue
                        groups[(submode, int(slot["slot"]), tier)] = np.stack(
                            [ordinals[mask].astype(np.float64), values[mask]], axis=1
                        )
        for (submode, slot_index, tier), entries in sorted(groups.items()):
            slot = TARGET_SLOTS[slot_index]
            required = required_by_key[(mode, submode, str(slot["name"]))]
            reports, summary = _verify_group(
                entries, mode, submode, slot, required, verification_dps, budget
            )
            for report in reports:
                report["tier"] = "core" if tier == 0 else "stress"
                report["mode"] = mode
            verified.extend(reports)
            verification_summary.append(
                {
                    "mode": mode,
                    "submode": submode,
                    "target": slot["name"],
                    "role": slot["role"],
                    "tier": "core" if tier == 0 else "stress",
                    "survivors": sum(1 for item in reports if item["survived_chance_gate"]),
                    **summary,
                }
            )

    # A program can be verified more than once -- as a core candidate, as a stress candidate,
    # and as a named control.  Survivors are keyed by the claim they make, which is
    # (mode, submode, program, target), so nothing is counted twice.
    unique_survivors: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for item in verified:
        if not item["survived_chance_gate"]:
            continue
        key = (
            str(item["mode"]),
            str(item["submode"]),
            int(item["ordinal"]),
            str(item["target"]),
        )
        if key not in unique_survivors or item["tier"] == "core":
            unique_survivors[key] = item
    # Every built-in classical identity inside the enumerated range is verified whether or
    # not the closest-first budget would have reached it, so the receipt shows the grammar
    # rediscovering known mathematics and the downstream reducing it.
    for item in BUILTIN_KNOWN_IDENTITIES:
        mode = str(item["mode"])
        ordinal = encode_program(_tokens_from_rpn(str(item["rpn"])), mode)
        if ordinal >= totals[mode]:
            continue
        slot = next(row for row in TARGET_SLOTS if row["name"] == item["target"])
        required = required_by_key[(mode, str(item["submode"]), str(item["target"]))]
        report = verify_candidate(
            ordinal, mode, str(item["submode"]), slot, required, verification_dps
        )
        report["tier"] = "control"
        report["mode"] = mode
        verified.append(report)
        if report["survived_chance_gate"]:
            unique_survivors[
                (mode, str(item["submode"]), ordinal, str(item["target"]))
            ] = report

    survivors = [unique_survivors[key] for key in sorted(unique_survivors)]
    decoy_survivors = [item for item in survivors if item["target_role"] == "decoy"]
    real_survivors = [item for item in survivors if item["target_role"] == "real"]

    # --- THE DECOY GATE: run-aborting ------------------------------------
    if decoy_survivors:
        names = sorted({str(item["target"]) for item in decoy_survivors})
        raise CompositionalSearchError(
            "decoy calibration failed: "
            f"{len(decoy_survivors)} decoy candidate(s) survived the precision gate at "
            f"{verification_dps} dps for {names}. The chance gate is too loose and this run "
            "is invalid."
        )

    # --- downstream: prior art, then classical reduction ------------------
    corpus_rows = None if skip_corpus else load_prior_art_values(root)
    builtin_by_ordinal = {
        (str(item["mode"]), encode_program(_tokens_from_rpn(str(item["rpn"])), str(item["mode"]))): item
        for item in BUILTIN_KNOWN_IDENTITIES
    }
    headline: list[dict[str, Any]] = []
    downstream: list[dict[str, Any]] = []
    for item in real_survivors:
        tokens = decode_ordinal(int(item["ordinal"]), str(item["mode"]))
        builtin = builtin_by_ordinal.get((str(item["mode"]), int(item["ordinal"])))
        prior_art = screen_prior_art(str(item["value"]), corpus_rows)
        reduction = classical_reduction(tokens, str(item["submode"]), str(item["target"]))
        record = {
            **item,
            "builtin_table": (
                None
                if builtin is None
                else {"id": builtin["id"], "attribution": builtin["attribution"]}
            ),
            "prior_art": prior_art,
            "classical_reduction": reduction,
        }
        if builtin is not None:
            record["verdict"] = "KNOWN_REDISCOVERED"
        elif reduction["verdict"] == "KNOWN_BY_PROOF_FAMILY":
            record["verdict"] = "KNOWN_BY_PROOF_FAMILY"
        elif prior_art["verdict"] == "CONSTANT_ATTESTED_IN_CORPUS":
            record["verdict"] = "UNREDUCED_CONSTANT_ATTESTED_IN_CORPUS"
        else:
            record["verdict"] = "UNREDUCED_AND_UNREVIEWED"
        downstream.append(record)
        if record["verdict"] == "UNREDUCED_AND_UNREVIEWED":
            headline.append(record)

    # --- C1: no null leaves this function without a reachability certificate ---
    # A headline of zero for a target is two different facts wearing one word -- the object does
    # not exist, or the grammar cannot spell it -- and until this block is attached the receipt
    # cannot tell a reader which one the sweep just produced.  The import is deferred because
    # the certifier reads this module's grammar; it is a hard dependency all the same, and a
    # campaign that cannot certify its own nulls does not finish.
    from .certified_null_search import campaign_reachability_block

    findings: dict[str, dict[str, int]] = {"C": {}, "F": {}}
    for record in headline:
        mode = str(record["mode"])
        target = str(record["target"])
        findings[mode][target] = findings[mode].get(target, 0) + 1
    reachability = campaign_reachability_block(findings)

    elapsed = time.perf_counter() - started
    enumerated_total = sum(totals.values())
    sweep_seconds = sum(
        float(sweeps[mode][pass_name]["elapsed_seconds"])
        for mode in ("C", "F")
        for pass_name in ("census", "retention")
    )
    measured_rate = enumerated_total * 2 / sweep_seconds if sweep_seconds > 0 else 0.0

    dedup: list[dict[str, Any]] = []
    for mode in ("C", "F"):
        engine = engines[mode]
        census = sweeps[mode]["census"]
        for index, submode in enumerate(engine.submodes):
            population = int(census["mag_hist"][index].sum())
            estimate = hll_estimate(census["hll"][index])
            dedup.append(
                {
                    "mode": mode,
                    "submode": submode,
                    "usable_values": population,
                    "distinct_value_estimate": format(estimate, ".6e"),
                    "distinct_over_evaluated": (
                        format(estimate / population, ".6e") if population else None
                    ),
                    "redundancy_factor": (
                        format(population / estimate, ".3f") if estimate > 0 else None
                    ),
                    "hash_precision": f"{VALUE_HASH_MANTISSA_BITS}-bit mantissa (~12 decimals)",
                    "estimator_standard_error": format(HLL_STANDARD_ERROR, ".4f"),
                }
            )
    controls["dedup_exact_crosscheck"] = [
        dedup_exact_control(engines["C"], totals["C"], DEDUP_EXACT_SAMPLE),
        dedup_exact_control(engines["F"], totals["F"], 1 << 23),
    ]
    if not all(row["passes"] for row in controls["dedup_exact_crosscheck"]):
        raise CompositionalSearchError(
            "the HyperLogLog distinct-value estimate did not survive its exact cross-check"
        )

    codec_validity = []
    for mode in ("C", "F"):
        closed_form = count_valid_programs(mode)
        status = sweeps[mode]["census"]["status_counts"]
        measured_valid = int(status[1])
        expected = (
            closed_form["structurally_valid"]
            if totals[mode] == int(MODE_CONFIG[mode]["space_size"])
            else None
        )
        codec_validity.append(
            {
                "mode": mode,
                "space_size": int(MODE_CONFIG[mode]["space_size"]),
                "enumerated": totals[mode],
                "structurally_valid_measured": measured_valid,
                "structurally_valid_closed_form": expected,
                "agrees_with_closed_form": None if expected is None else measured_valid == expected,
                "valid_fraction_measured": format(measured_valid / max(1, totals[mode]), ".8f"),
                "valid_with_variable_measured": int(status[5]) or None,
                "domain_or_overflow_rejected": int(status[2]),
                "usable_values": int(status[3]),
                "series_admitted": int(status[6]) or None,
                "limit_admitted": int(status[7]) or None,
            }
        )
    if any(row["agrees_with_closed_form"] is False for row in codec_validity):
        raise CompositionalSearchError(
            "codec control failed: the measured valid fraction disagrees with the "
            "closed-form count"
        )

    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "claims": CLAIMS,
        "decision": "SEARCHED",
        "device": device,
        "scope": (
            "Exhaustive brute-force enumeration of a generative RPN expression grammar over a "
            "declared 24-token alphabet: 1,801,152,661,463 length-9 constant programs and "
            "110,075,314,176 length-8 index programs, evaluated by a GPU stack machine with no "
            "semantic understanding and matched against a declared target list. Every match "
            "count is measured; the verification precision is derived from the measured local "
            "value density; decoy targets calibrate the false-positive rate empirically. "
            "Survivors are conjectures, not theorems, and an unreduced survivor is unreviewed, "
            "not novel."
        ),
        "grammar": {
            "generative": True,
            "tokens": [
                {"index": index, "name": name, "arity": TOKEN_ARITY[index]}
                for index, name in enumerate(TOKEN_NAMES)
            ],
            "token_count": TOKEN_COUNT,
            "terminal_count": TERMINAL_COUNT,
            "unary_count": len(UNARY_TOKENS),
            "binary_count": len(BINARY_TOKENS),
            "stack_depth": STACK_DEPTH,
            "value_cap": format(VALUE_CAP, ".0e"),
            "trig_argument_bound": format(TRIG_ARGUMENT_BOUND, ".0f"),
            "codec": (
                "mixed-radix little-endian over the mode alphabet: digit i of the ordinal is "
                "token i of the program, so the ordinal's least significant digit is the "
                "program's first instruction; an injection onto token sequences, made an "
                "injection onto programs by the validity pass"
            ),
            "validity_rule": (
                "reject stack underflow, reject a push past the declared stack depth, and "
                "reject any sequence that does not terminate with exactly one value"
            ),
            "domain_rule": (
                "recip of zero, sqrt of a negative, ln of a non-positive, division by zero, a "
                "fractional power of a negative base, sin or cos of an argument above "
                f"{TRIG_ARGUMENT_BOUND:.0f} (past which a double no longer determines the "
                "value to the match tolerance), a non-finite intermediate, or any intermediate "
                "above the value cap kills the program"
            ),
            "modes": {mode: MODE_CONFIG[mode] for mode in ("C", "F")},
            "series_start_index": SERIES_START_INDEX,
        },
        "codec_validity": codec_validity,
        "campaign": {
            "enumerated_total": enumerated_total,
            "enumerated_by_mode": totals,
            "exhaustive_over_declared_space": {
                mode: totals[mode] == int(MODE_CONFIG[mode]["space_size"]) for mode in ("C", "F")
            },
            "passes_per_mode": 2,
            "candidate_evaluations": enumerated_total * 2,
            "sweep_seconds": format(sweep_seconds, ".3f"),
            "measured_rate_ordinals_per_second": format(measured_rate, ".4e"),
            "rate_note": (
                "sweep_seconds is the accumulated kernel-and-launch time across every session "
                "that contributed to this result, including resumes, so the rate is a "
                "measurement of the whole enumeration and not of one session's tail"
            ),
            "resumed": {
                mode: {
                    pass_name: int(sweeps[mode][pass_name]["resumed_from_ordinal"])
                    for pass_name in ("census", "retention")
                }
                for mode in ("C", "F")
            },
            "seconds_to_1e12_at_measured_rate": (
                format(1e12 / measured_rate, ".1f") if measured_rate > 0 else None
            ),
            "reached_1e12": enumerated_total >= 10**12,
            "chunking": {
                mode: {
                    "chunk_size": chunk_size,
                    "launch_size": launch_size,
                    "chunks": sweeps[mode]["census"]["chunks"],
                    "census_lineage_sha256": sweeps[mode]["census"]["chunk_lineage_sha256"],
                    "retention_lineage_sha256": sweeps[mode]["retention"][
                        "chunk_lineage_sha256"
                    ],
                    "restart_safe": True,
                }
                for mode in ("C", "F")
            },
            "mode_f_stage2": {
                "near_window": format(MODE_F_NEAR_WINDOW, ".0e"),
                "stage1_terms": STAGE1_TERMS,
                "stage1_ladder": STAGE1_LADDER,
                "stage2_terms": STAGE2_TERMS,
                "stage2_ladder": STAGE2_LADDER,
                "extrapolation": (
                    "Richardson in 1/N over a geometric ladder of partial sums; a value is "
                    f"admitted only when the last two Richardson columns agree to "
                    f"{STAGE1_STABILITY:.0e} relative at stage 1 and {STAGE2_STABILITY:.0e} at "
                    "stage 2, so acceleration can cost recall but cannot manufacture a match"
                ),
                "stage1_stability": format(STAGE1_STABILITY, ".0e"),
                "stage2_stability": format(STAGE2_STABILITY, ".0e"),
                "near_target_ordinals": int(unique_f.size),
                "near_target_retention_complete": mode_f_complete,
                "retention_truncated": {
                    mode: bool(sweeps[mode]["retention"]["hits_truncated"])
                    for mode in ("C", "F")
                },
                "series_admitted": int(stage2["series_ordinals"].size),
                "limit_admitted": int(stage2["limit_ordinals"].size),
                "fine_windows_recomputed_below": f"1e-{STAGE1_WINDOW_FLOOR_EXPONENT}",
            },
        },
        "value_distribution": [
            {
                "mode": mode,
                "submode": submode,
                **_magnitude_summary(sweeps[mode]["census"]["mag_hist"][index]),
            }
            for mode in ("C", "F")
            for index, submode in enumerate(MODE_CONFIG[mode]["submodes"])
        ],
        "deduplication": dedup,
        "chance_gate": gate,
        "verification": {
            "verification_dps": verification_dps,
            "ladder": (
                "each candidate is re-evaluated at the derived precision and again at twice "
                "that precision; a survivor must clear two bars at both, the chance bar (its "
                "own derived agreement-digit requirement) and the holdout bar (agreement "
                f"within {AGREEMENT_GUARD_DIGITS} digits of the working precision). The "
                "holdout bar exists because asymptotic near-identities -- atan of a huge "
                "argument saturating at pi/2, for instance -- beat any chance model while "
                "their agreement stops growing with precision"
            ),
            "agreement_guard_digits": AGREEMENT_GUARD_DIGITS,
            "verify_budget_per_group": {"C": verify_budget_c, "F": verify_budget_f},
            "groups": verification_summary,
        },
        "decoy_calibration": _decoy_table(gate, survivors, verification_summary),
        "survivor_counts": {
            "fp64_matches_real": sum(
                int(row["fp64_matches_at_1e-12"])
                for mode in ("C", "F")
                for row in gate[mode]["per_target"]
                if row["role"] == "real"
            ),
            "fp64_matches_decoy": sum(
                int(row["fp64_matches_at_1e-12"])
                for mode in ("C", "F")
                for row in gate[mode]["per_target"]
                if row["role"] == "decoy"
            ),
            "verified_candidates": len(verified),
            "post_gate_survivors_real": len(real_survivors),
            "post_gate_survivors_decoy": len(decoy_survivors),
            "known_rediscovered": sum(
                1 for row in downstream if row["verdict"] == "KNOWN_REDISCOVERED"
            ),
            "known_by_proof_family": sum(
                1 for row in downstream if row["verdict"] == "KNOWN_BY_PROOF_FAMILY"
            ),
            "unreduced_constant_attested": sum(
                1
                for row in downstream
                if row["verdict"] == "UNREDUCED_CONSTANT_ATTESTED_IN_CORPUS"
            ),
            "headline_unreduced_and_unreviewed": len(headline),
        },
        "downstream": downstream,
        "reachability": reachability,
        "headline": {
            "count": len(headline),
            "meaning": (
                "candidates that beat the measured chance model at the derived precision, are "
                "absent from the prior-art corpus, and were not reduced by any declared "
                "classical technique. This is not a novelty claim: absence from a finite "
                "corpus establishes nothing, and an unreduced candidate is unreviewed rather "
                "than new."
            ),
            "entries": headline,
        },
        "controls": controls,
        "corpus": {
            "database": str(CORPUS_DATABASE),
            "records": None if corpus_rows is None else len(corpus_rows),
            "available": corpus_rows is not None,
            "value_match_digits": 40,
            "rule": (
                "a value hit reports that the CONSTANT is attested, never that the EXPRESSION "
                "is known"
            ),
        },
    }
    core = canonical_sha256(body)
    body["result_core_sha256"] = core
    body["measurement"] = {"elapsed_seconds": format(elapsed, ".3f")}
    result = {**body, "content_sha256": canonical_sha256(body)}
    # The run does not finish until its own receipt passes its own validator.
    validate_receipt(result)
    return result


def _decoy_table(
    gate: Mapping[str, Any],
    survivor_records: Sequence[Mapping[str, Any]],
    verification_summary: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The real column and the decoy column, side by side.  This is the calibration.

    Two windows appear here for a reason.  At the 1e-12 match window the decoys draw nothing
    at all, which is a strong result -- it says chance matches at the declared tolerance do not
    happen at this campaign size -- but it leaves the precision gate untested, because a gate
    that is never handed a chance match has not been shown to reject one.  The stress annulus
    [1e-14, 1e-8) is where the chance matches actually live: tens per decoy, thousands for some
    real targets.  Those are put through the same gate, and the decoy column of that row is the
    measurement that makes the real column mean something.
    """

    survivors: dict[tuple[str, str, str], int] = {}
    for row in survivor_records:
        key = (str(row["mode"]), str(row["submode"]), str(row["target"]))
        survivors[key] = survivors.get(key, 0) + 1
    verified: dict[tuple[str, str, str, str], dict[str, int]] = {}
    for row in verification_summary:
        key = (str(row["mode"]), str(row["submode"]), str(row["target"]), str(row["tier"]))
        entry = verified.setdefault(key, {"verified": 0, "survivors": 0})
        entry["verified"] += int(row["verified"])
        entry["survivors"] += int(row["survivors"])
    rows: list[dict[str, Any]] = []
    for mode in ("C", "F"):
        for row in gate[mode]["per_target"]:
            submode = str(row["submode"])
            name = str(row["target"])
            counts = row["cumulative_counts"]
            annulus = int(counts[f"1e-{STRESS_WINDOW_EXPONENT}"]) - int(
                counts[f"1e-{CORE_WINDOW_EXPONENT}"]
            )
            core = verified.get((mode, submode, name, "core"), {"verified": 0, "survivors": 0})
            stress = verified.get(
                (mode, submode, name, "stress"), {"verified": 0, "survivors": 0}
            )
            rows.append(
                {
                    "mode": mode,
                    "submode": submode,
                    "target": name,
                    "role": row["role"],
                    "paired_with": row["paired_with"],
                    "fp64_matches_at_1e-12": row["fp64_matches_at_1e-12"],
                    "expected_chance_matches": row["expected_chance_matches"],
                    "required_agreement_digits": row["required_agreement_digits"],
                    "stress_annulus_population": annulus,
                    "core_verified": core["verified"],
                    "core_survivors": core["survivors"],
                    "stress_verified": stress["verified"],
                    "stress_survivors": stress["survivors"],
                    "post_gate_survivors": survivors.get((mode, submode, name), 0),
                }
            )
    real = [row for row in rows if row["role"] == "real"]
    decoy = [row for row in rows if row["role"] == "decoy"]

    def total(block: Sequence[Mapping[str, Any]], key: str) -> int:
        return sum(int(row[key]) for row in block)

    return {
        "rule": (
            "decoys are reals with no known closed form drawn from a declared seed; each real "
            "target is paired with one sitting 1e-3..1e-2 away so it samples the same local "
            "value density. Decoys must attract chance matches and must produce zero survivors "
            "after the precision gate; a surviving decoy aborts the run."
        ),
        "seed": DECOY_SEED,
        "stress_window": format(STRESS_WINDOW, ".0e"),
        "core_window": format(CORE_WINDOW, ".0e"),
        "totals": {
            "real_fp64_matches": total(real, "fp64_matches_at_1e-12"),
            "decoy_fp64_matches": total(decoy, "fp64_matches_at_1e-12"),
            "real_stress_annulus_population": total(real, "stress_annulus_population"),
            "decoy_stress_annulus_population": total(decoy, "stress_annulus_population"),
            "real_stress_verified": total(real, "stress_verified"),
            "decoy_stress_verified": total(decoy, "stress_verified"),
            "real_stress_survivors": total(real, "stress_survivors"),
            "decoy_stress_survivors": total(decoy, "stress_survivors"),
            "real_post_gate_survivors": total(real, "post_gate_survivors"),
            "decoy_post_gate_survivors": total(decoy, "post_gate_survivors"),
        },
        "per_target": rows,
    }


# ---------------------------------------------------------------------------
# Receipt validation and CLI
# ---------------------------------------------------------------------------


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Seal, schema, claim, and gate checks.  A replay is a re-run, not a validation."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise CompositionalSearchError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise CompositionalSearchError("receipt seal changed")
    core = {
        key: item
        for key, item in body.items()
        if key not in {"result_core_sha256", "measurement"}
    }
    if body.get("result_core_sha256") != canonical_sha256(core):
        raise CompositionalSearchError("receipt core binding changed")
    for key, expected in CLAIMS.items():
        if value.get("claims", {}).get(key) != expected:
            raise CompositionalSearchError(f"claim {key} is missing or flipped")
    calibration = value.get("decoy_calibration", {}).get("totals", {})
    if int(calibration.get("decoy_post_gate_survivors", 1)) != 0:
        raise CompositionalSearchError("a decoy survived the precision gate; the run is invalid")
    for row in value.get("controls", {}).get("reachability", []):
        if not row.get("reproduces_target"):
            raise CompositionalSearchError(f"rediscovery control {row.get('id')} is not green")
    for row in value.get("controls", {}).get("determinism", []):
        if not (row.get("identical_accumulators") and row.get("identical_retained_ordinals")):
            raise CompositionalSearchError("determinism control is not green")
    for row in value.get("controls", {}).get("cpu_gpu_crosscheck", []):
        if not row.get("passes"):
            raise CompositionalSearchError("CPU/GPU cross-check control is not green")
    for row in value.get("controls", {}).get("dedup_exact_crosscheck", []):
        if not row.get("passes"):
            raise CompositionalSearchError("dedup exact cross-check control is not green")
    tamper = value.get("controls", {}).get("receipt_tamper")
    if tamper is not None and not (
        tamper.get("untampered_receipt_validates") and tamper.get("all_probes_rejected")
    ):
        raise CompositionalSearchError("receipt tamper control is not green")
    for row in value.get("codec_validity", []):
        if row.get("agrees_with_closed_form") is False:
            raise CompositionalSearchError("measured valid fraction disagrees with the codec")
    for entry in value.get("headline", {}).get("entries", []):
        if not entry.get("survived_chance_gate"):
            raise CompositionalSearchError("a headline entry did not survive the chance gate")
        if entry.get("classical_reduction", {}).get("verdict") == "KNOWN_BY_PROOF_FAMILY":
            raise CompositionalSearchError("a reduced candidate is in the headline")
    _validate_reachability(value)


def _validate_reachability(value: Mapping[str, Any]) -> None:
    """The C1 gate: a null may not be published without a reachability certificate.

    Two jobs.  A receipt that reports nothing and carries no certificate is refused outright --
    "we searched 1.9e12 expressions and found nothing" is indistinguishable from "we searched
    the wrong 1.9e12 expressions" until something proves the target was in the space.  And a
    receipt that *does* carry a block has that block re-verified from scratch: every certificate
    re-proved, every outcome re-derived, the block regenerated and hash-compared, and the
    block's own findings reconciled against the receipt's own headline entries, so a block
    cannot be attached that adjudicates a campaign other than the one it is stapled to.
    """

    # The import is deferred for the same reason it is deferred in ``run_campaign``: the
    # certifier is a consumer of this module's grammar.
    from .certified_null_search import (
        OUTCOME_INCONSISTENT,
        OUTCOME_REAL_NEGATIVE,
        verify_campaign_reachability_block,
    )

    block = value.get("reachability")
    headline_count = int(value.get("headline", {}).get("count", 0))
    if block is None:
        if headline_count == 0:
            raise CompositionalSearchError(
                "C1: this receipt reports a null -- zero headline candidates -- and carries no "
                "reachability certificate, so it cannot distinguish 'no such object exists' "
                "from 'the grammar could not express it'. An uninformative null is not a result"
            )
        return

    verify_campaign_reachability_block(block)
    rows = {
        (str(row["mode"]), str(row["target"])): row
        for row in block.get("null_adjudication", [])
    }
    searched = {
        (mode, str(row["target"]))
        for mode in ("C", "F")
        for row in value.get("chance_gate", {}).get(mode, {}).get("per_target", [])
        if row.get("role") == "real"
    }
    missing = sorted(f"{mode}/{target}" for mode, target in searched - set(rows))
    if missing:
        raise CompositionalSearchError(
            f"C1: these targets were searched but carry no reachability adjudication: {missing}"
        )
    counted: dict[tuple[str, str], int] = {}
    for entry in value.get("headline", {}).get("entries", []):
        key = (str(entry.get("mode")), str(entry.get("target")))
        counted[key] = counted.get(key, 0) + 1
    for key, row in rows.items():
        if int(row["findings"]) != counted.get(key, 0):
            raise CompositionalSearchError(
                f"C1: the reachability block reports {row['findings']} finding(s) for "
                f"{key[0]}/{key[1]} but the receipt's own headline carries "
                f"{counted.get(key, 0)}"
            )
        if row["outcome"] == OUTCOME_INCONSISTENT:
            raise CompositionalSearchError(
                f"C1: the reachability certificate for {key[0]}/{key[1]} disagrees with the "
                "campaign's own finding count"
            )
        if row["outcome"] == OUTCOME_REAL_NEGATIVE and not row.get("certificate_verified"):
            raise CompositionalSearchError(
                f"C1: {key[0]}/{key[1]} is published as a real negative on an unverified "
                "certificate"
            )


def _write_receipt(result: Mapping[str, Any], output: str) -> None:
    path = Path(output)
    encoded = canonical_json_bytes(result) + b"\n"
    if path.exists() and path.read_bytes() != encoded:
        raise CompositionalSearchError("refusing to overwrite immutable receipt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trillion-scale compositional expression search with a calibrated "
        "chance-match gate."
    )
    parser.add_argument("--root", default=".", help="repository root (for the prior-art corpus)")
    parser.add_argument("--mode-c-limit", type=int, default=None)
    parser.add_argument("--mode-f-limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--launch-size", type=int, default=DEFAULT_LAUNCH_SIZE)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--verify-budget-c", type=int, default=VERIFY_BUDGET_PER_SLOT)
    parser.add_argument("--verify-budget-f", type=int, default=32)
    parser.add_argument("--skip-corpus", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--validate-checked", action="store_true")
    parser.add_argument("--print-alphabet", action="store_true")
    arguments = parser.parse_args()

    if arguments.print_alphabet:
        print(
            json.dumps(
                {
                    "tokens": [
                        {"index": index, "name": name, "arity": TOKEN_ARITY[index]}
                        for index, name in enumerate(TOKEN_NAMES)
                    ],
                    "token_count": TOKEN_COUNT,
                    "stack_depth": STACK_DEPTH,
                    "modes": {
                        mode: {
                            "program_length": MODE_CONFIG[mode]["program_length"],
                            "alphabet_size": MODE_CONFIG[mode]["alphabet_size"],
                            "space_size": MODE_CONFIG[mode]["space_size"],
                            **count_valid_programs(mode),
                        }
                        for mode in ("C", "F")
                    },
                },
                indent=2,
            )
        )
        return 0
    if arguments.validate_checked:
        if not arguments.output:
            raise CompositionalSearchError("--validate-checked needs --output")
        validate_receipt(json.loads(Path(arguments.output).read_text(encoding="utf-8")))
        return 0

    result = run_campaign(
        root=Path(arguments.root),
        mode_c_limit=arguments.mode_c_limit,
        mode_f_limit=arguments.mode_f_limit,
        chunk_size=arguments.chunk_size,
        launch_size=arguments.launch_size,
        checkpoint_dir=(
            None if arguments.checkpoint_dir is None else Path(arguments.checkpoint_dir)
        ),
        verify_budget_c=arguments.verify_budget_c,
        verify_budget_f=arguments.verify_budget_f,
        skip_corpus=arguments.skip_corpus,
    )
    if arguments.output:
        _write_receipt(result, arguments.output)
    print(
        json.dumps(
            {
                "enumerated_total": result["campaign"]["enumerated_total"],
                "measured_rate_ordinals_per_second": result["campaign"][
                    "measured_rate_ordinals_per_second"
                ],
                "verification_dps": result["verification"]["verification_dps"],
                "max_required_agreement_digits": max(
                    result["chance_gate"][mode]["max_required_agreement_digits"]
                    for mode in ("C", "F")
                ),
                "decoy_calibration": result["decoy_calibration"]["totals"],
                "survivor_counts": result["survivor_counts"],
                "headline": result["headline"]["count"],
                "device": result["device"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
