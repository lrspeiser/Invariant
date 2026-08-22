"""Unlabelled channels with a description-length price, so a dependence can be *discovered*.

The engine's proposal loop declares a problem as ``rule(u) -> value``: one number in, one
number out.  Anything that needs a second reading -- a second measured column, whatever it
happens to be -- is not merely hard to find, it is **outside the search space**.  A search
cannot discover a dependence on a quantity it has no slot for.

This module supplies the slot and the price.

**The slot.**  A :class:`ChannelTable` is a set of rows of exact rationals.  Each row has
``arity`` entries and they are addressed *by index*.  They are never named after anything.
The rendered symbols are ``u``, ``w``, ``x`` and then ``c3``, ``c4``, ... -- placeholders
chosen because they say nothing.  Naming a channel would hand the proposer the concept it is
supposed to find, which is exactly the blindness failure the rest of the repository guards
against, so the naming is refused at declaration time rather than discouraged in prose.

**The price.**  A law is scored by a genuine two-part minimum-description-length code:

    total_bits = model_bits + admission_bits + data_bits

``model_bits`` is the length of a **decodable** prefix-free encoding of the law itself
(:func:`encode_expression` / :func:`decode_expression`).  ``data_bits`` is the length of a
**decodable** prefix-free encoding of the quantised residuals
(:func:`encode_residuals` / :func:`decode_observations`), which reconstructs every observation
to within half the declared resolution.  The two together are a message from which a receiver
holding only the table's channel readings recovers the observations.  That is what makes the
number a description length rather than a penalty someone invented: it is the length of a
message that actually decodes, and :mod:`tests.test_unlabelled_channel_mdl` decodes it.

``admission_bits`` is the declared price of *widening the law's domain*: a flat fee charged
once per channel the law is measured to depend on.  It is declared before the data, sealed in
:class:`ChannelPolicy`, and hashed into the receipt.  A channel that does not repay its fee in
residual bits is not adopted, however well it appears to help.

**Dependence is measured, not read.**  Whether a law uses a channel is decided *behaviourally*:
the table is a complete factorial design, so for any channel there are pairs of rows differing
in that coordinate alone, and the law is evaluated on both.  A law that writes ``0 * x`` and a
law that never mentions ``x`` are the same function and are classified the same way -- but the
first pays for the tokens it wrote, because that is what description length means.  On a design
with no such witness pair the answer is ``UNDECIDED`` and the law is set aside; the module does
not guess and then call the guess a measurement.

**The obstruction is policy-free and universally quantified.**  Separately from any law space,
:func:`ignoring_codelength_floor` computes an exact integer ``F`` such that *no function
whatsoever* that ignores the declared channels can encode the observations in fewer than ``F``
bits at the declared resolution.  The argument is elementary and exact: a function ignoring
channel ``j`` must emit one prediction for every row of a group that differs only in ``j``, two
rows of that group whose observations differ by ``D`` force a quantised residual of magnitude at
least ``ceil((ceil(D/resolution) - 1)/2)`` on one of them, and disjoint pairs add.  So when an
exhibited law that *uses* channel ``j`` codes the same data in fewer bits than ``F``, the
sentence "no function of the other channels alone reaches this quality" is a theorem about the
table, not a statement about how hard the search tried.

**Everything on the certificate path is exact.**  Channel readings, observations, resolutions,
predictions and residuals are :class:`fractions.Fraction`; codelengths are :class:`int`; every
comparison is an integer or rational comparison.  There is no floating point anywhere in this
module.  The one place floats appear in the wider system is a sandboxed program's output, and
:func:`exact_predictions_from_outputs` converts those decimal strings to exact rationals at the
boundary before any arithmetic happens.

**What this is not.**  ``adoption`` is a decision under a declared policy and it says so; move
the fee and the decision can move, which is why the receipt publishes the exact fee at which
each verdict flips.  ``obstruction`` carries no policy and does not move.  And a law space is
finite, so "the best law that ignores channel ``j``" is a statement about the declared space --
the universally quantified statement is the obstruction, and only the obstruction.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

RECEIPT_SCHEMA = "invariant-unlabelled-channel-mdl-1.0"
CAMPAIGN_ID = "unlabelled-channel-mdl-001"

CLAIMS = {
    "channel_dependence_is_measured_behaviourally": True,
    "channel_names_carry_no_meaning": True,
    "every_certificate_number_is_an_exact_integer": True,
    "obstruction_is_universally_quantified_over_functions": True,
    "using_a_channel_is_paid_for": True,
}

SCOPE = (
    "An exact two-part minimum-description-length account of whether an unlabelled measured "
    "channel earns its place. Channels are addressed by index and rendered with placeholder "
    "symbols; a declaration that names one is refused. Model bits and data bits are the "
    "lengths of decodable prefix-free codes, so the total is a message length rather than a "
    "penalty. The adoption verdict is a decision under a sealed policy and moves with it; the "
    "obstruction floor carries no policy and is a lower bound over all functions ignoring the "
    "declared channels."
)


class ChannelMDLError(ValueError):
    """Raised on a malformed table, a named channel, or a receipt that no longer seals."""


# ---------------------------------------------------------------------------
# 0. Placeholder symbols.  These are the whole of the naming discipline.
# ---------------------------------------------------------------------------

#: The symbols a channel may be rendered with.  ``u``, ``w`` and ``x`` match the widened
#: signature in :mod:`.funsearch_loop`; beyond three channels the symbol is the bare index, so
#: it is impossible to accidentally choose an evocative one.
OPAQUE_CHANNEL_SYMBOLS = ("u", "w", "x")


def channel_symbol(index: int) -> str:
    """The placeholder for channel ``index``.  Deliberately says nothing."""

    if index < 0:
        raise ChannelMDLError(f"channel index is negative: {index}")
    if index < len(OPAQUE_CHANNEL_SYMBOLS):
        return OPAQUE_CHANNEL_SYMBOLS[index]
    return f"c{index}"


# ---------------------------------------------------------------------------
# 1. Exact integer codes.  Every codelength in this module comes from here.
# ---------------------------------------------------------------------------


def _ceil_fraction(value: Fraction) -> int:
    """``ceil`` of an exact rational, by integer division only."""

    return -((-value.numerator) // value.denominator)


def _round_half_up(value: Fraction) -> int:
    """``floor(value + 1/2)`` in integer arithmetic.  The declared quantiser."""

    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def gamma_bits(magnitude: int) -> int:
    """Elias-gamma length of ``magnitude >= 1``: exact, and exactly ``2*floor(log2 m) + 1``."""

    if magnitude < 1:
        raise ChannelMDLError(f"gamma code is defined for positive integers: {magnitude}")
    return 2 * magnitude.bit_length() - 1


def encode_gamma(magnitude: int) -> str:
    """``magnitude`` as an Elias-gamma bit string."""

    if magnitude < 1:
        raise ChannelMDLError(f"gamma code is defined for positive integers: {magnitude}")
    body = bin(magnitude)[3:]
    return "0" * (magnitude.bit_length() - 1) + "1" + body


def decode_gamma(bits: str, position: int) -> tuple[int, int]:
    """Read one Elias-gamma codeword from ``bits`` at ``position``."""

    zeros = 0
    while position + zeros < len(bits) and bits[position + zeros] == "0":
        zeros += 1
    position += zeros
    if position >= len(bits) or bits[position] != "1":
        raise ChannelMDLError("truncated gamma codeword")
    position += 1
    magnitude = 1
    for _ in range(zeros):
        if position >= len(bits):
            raise ChannelMDLError("truncated gamma payload")
        magnitude = magnitude * 2 + (1 if bits[position] == "1" else 0)
        position += 1
    return magnitude, position


def integer_code_bits(value: int) -> int:
    """Length of the declared signed code for ``value``.  Nondecreasing in ``abs(value)``.

    ``gamma(abs(value) + 1)``, plus one sign bit when the value is nonzero.  Zero costs one
    bit, which is the floor every row pays whatever law is fitted.
    """

    return gamma_bits(abs(value) + 1) + (1 if value else 0)


def encode_integer(value: int) -> str:
    bits = encode_gamma(abs(value) + 1)
    if value:
        bits += "1" if value < 0 else "0"
    return bits


def decode_integer(bits: str, position: int) -> tuple[int, int]:
    magnitude, position = decode_gamma(bits, position)
    value = magnitude - 1
    if value == 0:
        return 0, position
    if position >= len(bits):
        raise ChannelMDLError("truncated sign bit")
    negative = bits[position] == "1"
    return (-value if negative else value), position + 1


# ---------------------------------------------------------------------------
# 2. The law language.  Exact rationals in, exact rationals out.
# ---------------------------------------------------------------------------

#: Node kinds and their arities.  Three bits addresses eight, one of which is left unused so
#: the tag width is honest rather than fitted.
NODE_ARITY: dict[str, int] = {
    "chan": 0,
    "const": 0,
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "neg": 1,
}
NODE_TAGS = tuple(NODE_ARITY)
NODE_TAG_BITS = 3

#: A fixed channel-index width, independent of the table's arity.  This is what makes a law
#: that ignores a channel cost exactly what the same law costs on a narrower table: widening
#: the problem must be free for anything that does not use the width.
CHANNEL_INDEX_BITS = 4
MAX_CHANNELS = 1 << CHANNEL_INDEX_BITS


@dataclass(frozen=True, slots=True)
class Expr:
    """One node of a law.  Channels appear only as indices."""

    tag: str
    channel: int = -1
    value: Fraction = Fraction(0)
    children: tuple[Expr, ...] = ()

    def __post_init__(self) -> None:
        if self.tag not in NODE_ARITY:
            raise ChannelMDLError(f"undeclared node tag: {self.tag}")
        if len(self.children) != NODE_ARITY[self.tag]:
            raise ChannelMDLError(f"node {self.tag} takes {NODE_ARITY[self.tag]} children")
        if self.tag == "chan" and not 0 <= self.channel < MAX_CHANNELS:
            raise ChannelMDLError(f"channel index outside the declared width: {self.channel}")


def chan(index: int) -> Expr:
    return Expr("chan", channel=index)


def const(value: Fraction | int) -> Expr:
    return Expr("const", value=Fraction(value))


def add(left: Expr, right: Expr) -> Expr:
    return Expr("add", children=(left, right))


def sub(left: Expr, right: Expr) -> Expr:
    return Expr("sub", children=(left, right))


def mul(left: Expr, right: Expr) -> Expr:
    return Expr("mul", children=(left, right))


def div(left: Expr, right: Expr) -> Expr:
    return Expr("div", children=(left, right))


def neg(operand: Expr) -> Expr:
    return Expr("neg", children=(operand,))


def referenced_channels(expr: Expr) -> tuple[int, ...]:
    """The channel indices the law *writes down*, whether or not it depends on them."""

    found: set[int] = set()

    def walk(node: Expr) -> None:
        if node.tag == "chan":
            found.add(node.channel)
        for child in node.children:
            walk(child)

    walk(expr)
    return tuple(sorted(found))


def evaluate(expr: Expr, row: Sequence[Fraction]) -> Fraction | None:
    """Exact value of ``expr`` on ``row``, or ``None`` if the law is undefined there."""

    if expr.tag == "chan":
        if expr.channel >= len(row):
            raise ChannelMDLError(
                f"law reads channel {expr.channel} of a row with {len(row)} channels"
            )
        return row[expr.channel]
    if expr.tag == "const":
        return expr.value
    values = [evaluate(child, row) for child in expr.children]
    if any(item is None for item in values):
        return None
    if expr.tag == "neg":
        return -values[0]
    left, right = values[0], values[1]
    if expr.tag == "add":
        return left + right
    if expr.tag == "sub":
        return left - right
    if expr.tag == "mul":
        return left * right
    if right == 0:
        return None
    return left / right


def expression_text(expr: Expr) -> str:
    """A hand-checkable rendering in placeholder symbols only."""

    if expr.tag == "chan":
        return channel_symbol(expr.channel)
    if expr.tag == "const":
        value = expr.value
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    if expr.tag == "neg":
        return f"-({expression_text(expr.children[0])})"
    operator = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[expr.tag]
    left = expression_text(expr.children[0])
    right = expression_text(expr.children[1])
    return f"({left} {operator} {right})"


def encode_expression(expr: Expr) -> str:
    """A prefix-free preorder encoding of the law.  ``model_bits`` is this string's length."""

    tag_index = NODE_TAGS.index(expr.tag)
    bits = format(tag_index, f"0{NODE_TAG_BITS}b")
    if expr.tag == "chan":
        return bits + format(expr.channel, f"0{CHANNEL_INDEX_BITS}b")
    if expr.tag == "const":
        value = expr.value
        bits += "1" if value.numerator < 0 else "0"
        bits += encode_gamma(abs(value.numerator) + 1)
        bits += encode_gamma(value.denominator)
        return bits
    for child in expr.children:
        bits += encode_expression(child)
    return bits


def decode_expression(bits: str, position: int = 0) -> tuple[Expr, int]:
    """Read one law back out of :func:`encode_expression`'s output."""

    if position + NODE_TAG_BITS > len(bits):
        raise ChannelMDLError("truncated node tag")
    tag_index = int(bits[position : position + NODE_TAG_BITS], 2)
    position += NODE_TAG_BITS
    if tag_index >= len(NODE_TAGS):
        raise ChannelMDLError(f"reserved node tag: {tag_index}")
    tag = NODE_TAGS[tag_index]
    if tag == "chan":
        if position + CHANNEL_INDEX_BITS > len(bits):
            raise ChannelMDLError("truncated channel index")
        index = int(bits[position : position + CHANNEL_INDEX_BITS], 2)
        return chan(index), position + CHANNEL_INDEX_BITS
    if tag == "const":
        if position >= len(bits):
            raise ChannelMDLError("truncated constant sign")
        negative = bits[position] == "1"
        position += 1
        magnitude, position = decode_gamma(bits, position)
        denominator, position = decode_gamma(bits, position)
        numerator = magnitude - 1
        return const(Fraction(-numerator if negative else numerator, denominator)), position
    children: list[Expr] = []
    for _ in range(NODE_ARITY[tag]):
        child, position = decode_expression(bits, position)
        children.append(child)
    return Expr(tag, children=tuple(children)), position


def model_bits(expr: Expr) -> int:
    return len(encode_expression(expr))


#: A lower bound on the model cost of any law in this language: one node, one tag.
MIN_MODEL_BITS = NODE_TAG_BITS


# ---------------------------------------------------------------------------
# 3. The table.  Rows of exact rationals, addressed by index.
# ---------------------------------------------------------------------------


def _rational_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True, slots=True)
class ChannelTable:
    """Exact readings, exact observations, and the declared coding resolution per row.

    ``resolutions`` is part of the sealed declaration, not derived from any law, so the
    residual code is decodable by a receiver holding the table and the law and nothing else.
    """

    table_id: str
    arity: int
    rows: tuple[tuple[Fraction, ...], ...]
    observations: tuple[Fraction, ...]
    resolutions: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if self.arity < 1 or self.arity > MAX_CHANNELS:
            raise ChannelMDLError(f"table arity outside the declared width: {self.arity}")
        if not self.rows:
            raise ChannelMDLError("a table needs at least one row")
        if len(self.rows) != len(self.observations) or len(self.rows) != len(self.resolutions):
            raise ChannelMDLError("rows, observations and resolutions must have equal length")
        if any(len(row) != self.arity for row in self.rows):
            raise ChannelMDLError("every row must carry exactly `arity` channel readings")
        if any(item <= 0 for item in self.resolutions):
            raise ChannelMDLError("every declared resolution must be positive")

    def contexts(self, dropped: Sequence[int]) -> dict[tuple[Fraction, ...], list[int]]:
        """Row indices grouped by everything except the ``dropped`` channels."""

        kept = [index for index in range(self.arity) if index not in set(dropped)]
        groups: dict[tuple[Fraction, ...], list[int]] = {}
        for position, row in enumerate(self.rows):
            key = tuple(row[index] for index in kept)
            groups.setdefault(key, []).append(position)
        return groups

    def witness_pairs(self, channel: int) -> tuple[tuple[int, int], ...]:
        """Row pairs agreeing everywhere except ``channel``, and *differing* there.

        The second half of that sentence is what stops a duplicated row from being mistaken
        for evidence: two identical rows can never separate a law's behaviour, so counting
        them would let an undecidable question be answered ``INDEPENDENT``.
        """

        if not 0 <= channel < self.arity:
            raise ChannelMDLError(f"channel {channel} is outside the table's arity {self.arity}")
        pairs: list[tuple[int, int]] = []
        for members in self.contexts((channel,)).values():
            for left in range(len(members)):
                for right in range(left + 1, len(members)):
                    first, second = members[left], members[right]
                    if self.rows[first][channel] != self.rows[second][channel]:
                        pairs.append((first, second))
        return tuple(pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "arity": self.arity,
            "channel_symbols": [channel_symbol(index) for index in range(self.arity)],
            "rows": [[_rational_text(item) for item in row] for row in self.rows],
            "observations": [_rational_text(item) for item in self.observations],
            "resolutions": [_rational_text(item) for item in self.resolutions],
        }

    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def collapsed_table(
    table: ChannelTable, keep: Sequence[int], table_id: str | None = None
) -> ChannelTable:
    """The same observations seen through a narrower slot: the channels ``keep``, in order.

    Collapsing renumbers channels, so a law is comparable across the wide and narrow tables
    only when ``keep`` leaves the channels it reads where they were.  Used with ``keep=(0,)``
    it answers the question the widening has to answer: does adding channels a law never uses
    cost that law anything?
    """

    keep = tuple(keep)
    if not keep or any(not 0 <= index < table.arity for index in keep):
        raise ChannelMDLError(f"cannot keep channels {keep} of an arity-{table.arity} table")
    return ChannelTable(
        table_id=table_id or f"{table.table_id}-keep-{'-'.join(str(item) for item in keep)}",
        arity=len(keep),
        rows=tuple(tuple(row[index] for index in keep) for row in table.rows),
        observations=table.observations,
        resolutions=table.resolutions,
    )


def factorial_table(
    table_id: str,
    levels: Sequence[Sequence[Fraction]],
    generator,
    resolution: Fraction,
) -> ChannelTable:
    """A complete factorial design over ``levels``, sealed against ``generator``.

    The design is complete on purpose: it is what makes "does this law depend on channel j"
    a decidable question about the table rather than an assumption.
    """

    rows: list[tuple[Fraction, ...]] = [()]
    for column in levels:
        rows = [row + (value,) for row in rows for value in column]
    ordered = tuple(sorted(rows))
    observations = tuple(Fraction(generator(row)) for row in ordered)
    return ChannelTable(
        table_id=table_id,
        arity=len(levels),
        rows=ordered,
        observations=observations,
        resolutions=tuple(resolution for _ in ordered),
    )


# ---------------------------------------------------------------------------
# 4. The policy.  Declared before the data and hashed into the receipt.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChannelPolicy:
    """The declared price list.

    ``channel_admission_bits`` is the flat fee for widening a law's domain by one measured
    channel.  It is a declared prior, not a measurement, and the receipt publishes the exact
    fee at which every adoption verdict flips so that its influence is visible rather than
    buried.  ``certification_margin_bits`` is how far under the universal obstruction floor an
    exhibited law must come before the module will say the channel is load bearing.
    """

    channel_admission_bits: int = 128
    adoption_margin_bits: int = 1
    certification_margin_bits: int = 16

    def __post_init__(self) -> None:
        if self.channel_admission_bits < 0:
            raise ChannelMDLError("the admission fee may not be negative")
        if self.adoption_margin_bits < 0 or self.certification_margin_bits < 0:
            raise ChannelMDLError("a margin may not be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_admission_bits": self.channel_admission_bits,
            "adoption_margin_bits": self.adoption_margin_bits,
            "certification_margin_bits": self.certification_margin_bits,
            "adoption_rule": (
                "adopted when (best total ignoring the channel) - (best total using it), with "
                "the admission fee excluded from both, is at least "
                "channel_admission_bits + adoption_margin_bits"
            ),
            "certification_rule": (
                "certified when an exhibited law using the channel codes the observations in "
                "at most (obstruction floor - certification_margin_bits) data bits"
            ),
        }

    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


DECLARED_POLICY = ChannelPolicy()


# ---------------------------------------------------------------------------
# 5. Codelength of one law on one table
# ---------------------------------------------------------------------------

DEPENDS = "DEPENDS"
INDEPENDENT = "INDEPENDENT"
UNDECIDED = "UNDECIDED"


@dataclass(frozen=True, slots=True)
class Dependence:
    verdict: str
    witness_pairs: int
    witness: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "witness_pairs_available": self.witness_pairs,
            "witness_rows": list(self.witness) if self.witness else None,
        }


def channel_dependence(table: ChannelTable, expr: Expr, channel: int) -> Dependence:
    """Does ``expr`` depend on ``channel``?  Measured on the table, never read off the source."""

    pairs = table.witness_pairs(channel)
    if not pairs:
        return Dependence(UNDECIDED, 0, None)
    for left, right in pairs:
        first = evaluate(expr, table.rows[left])
        second = evaluate(expr, table.rows[right])
        if first is None or second is None:
            continue
        if first != second:
            return Dependence(DEPENDS, len(pairs), (left, right))
    return Dependence(INDEPENDENT, len(pairs), None)


def used_channels(table: ChannelTable, expr: Expr) -> tuple[int, ...]:
    """The channels the law is *measured* to depend on.  ``UNDECIDED`` counts as used."""

    used: list[int] = []
    for index in range(table.arity):
        verdict = channel_dependence(table, expr, index).verdict
        if verdict in (DEPENDS, UNDECIDED):
            used.append(index)
    return tuple(used)


def predictions(table: ChannelTable, expr: Expr) -> tuple[Fraction, ...] | None:
    values: list[Fraction] = []
    for row in table.rows:
        value = evaluate(expr, row)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def residual_codes(
    table: ChannelTable, predicted: Sequence[Fraction]
) -> tuple[int, ...]:
    """The exact quantised residuals.  Integer arithmetic on exact rationals throughout."""

    if len(predicted) != len(table.observations):
        raise ChannelMDLError("prediction vector does not match the table")
    return tuple(
        _round_half_up((observed - value) / resolution)
        for observed, value, resolution in zip(
            table.observations, predicted, table.resolutions, strict=True
        )
    )


def encode_residuals(codes: Sequence[int]) -> str:
    return "".join(encode_integer(code) for code in codes)


def decode_observations(
    bits: str, table: ChannelTable, predicted: Sequence[Fraction]
) -> tuple[Fraction, ...]:
    """Reconstruct the observations from the residual message.  The proof the code is a code."""

    position = 0
    recovered: list[Fraction] = []
    for value, resolution in zip(predicted, table.resolutions, strict=True):
        code, position = decode_integer(bits, position)
        recovered.append(value + code * resolution)
    if position != len(bits):
        raise ChannelMDLError("residual message has trailing bits")
    return tuple(recovered)


def data_bits(codes: Sequence[int]) -> int:
    return sum(integer_code_bits(code) for code in codes)


@dataclass(frozen=True, slots=True)
class CodeLength:
    law_id: str
    model_bits: int
    admission_bits: int
    data_bits: int
    used_channels: tuple[int, ...]

    @property
    def total_bits(self) -> int:
        return self.model_bits + self.admission_bits + self.data_bits

    @property
    def unpriced_bits(self) -> int:
        """Total with the admission fee removed -- the quantity the fee is compared against."""

        return self.model_bits + self.data_bits

    def to_dict(self) -> dict[str, Any]:
        return {
            "law_id": self.law_id,
            "model_bits": self.model_bits,
            "admission_bits": self.admission_bits,
            "data_bits": self.data_bits,
            "total_bits": self.total_bits,
            "unpriced_bits": self.unpriced_bits,
            "used_channels": list(self.used_channels),
            "used_channel_symbols": [channel_symbol(item) for item in self.used_channels],
        }


def codelength(
    table: ChannelTable, law_id: str, expr: Expr, policy: ChannelPolicy = DECLARED_POLICY
) -> CodeLength | None:
    """The two-part codelength of ``expr`` on ``table``, or ``None`` if the law is undefined."""

    predicted = predictions(table, expr)
    if predicted is None:
        return None
    used = used_channels(table, expr)
    return CodeLength(
        law_id=law_id,
        model_bits=model_bits(expr),
        admission_bits=policy.channel_admission_bits * len(used),
        data_bits=data_bits(residual_codes(table, predicted)),
        used_channels=used,
    )


# ---------------------------------------------------------------------------
# 6. The obstruction floor.  No law space, no policy, no floats.
# ---------------------------------------------------------------------------


def pair_floor_bits(spread: Fraction, resolution: Fraction) -> int:
    """Bits forced on a pair of rows an ignoring function must predict identically.

    A function that ignores the dropped channels emits one value ``p`` for both rows.  With
    ``q = floor(r/resolution + 1/2)`` the quantiser moves each residual by less than half a
    step, so ``abs(q_a - q_b) >= ceil(spread/resolution - 1)``, and one of the two magnitudes
    is at least half of that.  The other row still costs the one-bit floor.
    """

    if spread < 0:
        raise ChannelMDLError("a spread may not be negative")
    separation = _ceil_fraction(spread / resolution - 1)
    if separation < 1:
        return integer_code_bits(0) * 2
    forced = -((-separation) // 2)
    return integer_code_bits(forced) + integer_code_bits(0)


def ignoring_codelength_floor(table: ChannelTable, dropped: Sequence[int]) -> dict[str, Any]:
    """An exact lower bound on the data bits of **any** function ignoring ``dropped``.

    Rows are grouped by the surviving coordinates; within a group the extremes are paired
    inward, which keeps the pairs disjoint so their forced bits add.
    """

    dropped = tuple(sorted(set(dropped)))
    for index in dropped:
        if not 0 <= index < table.arity:
            raise ChannelMDLError(f"channel {index} is outside the table's arity {table.arity}")
    if not dropped:
        raise ChannelMDLError("an obstruction must drop at least one channel")
    total = 0
    pairs_used = 0
    forced_groups = 0
    mixed_resolution_groups = 0
    for members in table.contexts(dropped).values():
        ordered = sorted(members, key=lambda position: table.observations[position])
        group_resolutions = {table.resolutions[position] for position in ordered}
        if len(group_resolutions) != 1:
            # The pair argument compares two residuals through one common quantiser step.  With
            # two step sizes in one group it does not apply, so the group falls back to the
            # bit every row pays whatever is fitted.  Refusing the bound is the honest move.
            mixed_resolution_groups += 1
            total += len(ordered) * integer_code_bits(0)
            continue
        resolution = next(iter(group_resolutions))
        low, high = 0, len(ordered) - 1
        group_bits = 0
        group_forced = False
        while low < high:
            spread = (
                table.observations[ordered[high]] - table.observations[ordered[low]]
            )
            bits = pair_floor_bits(spread, resolution)
            group_bits += bits
            pairs_used += 1
            if bits > 2 * integer_code_bits(0):
                group_forced = True
            low += 1
            high -= 1
        if low == high:
            group_bits += integer_code_bits(0)
        total += group_bits
        if group_forced:
            forced_groups += 1
    return {
        "dropped_channels": list(dropped),
        "dropped_channel_symbols": [channel_symbol(item) for item in dropped],
        "groups": len(table.contexts(dropped)),
        "disjoint_pairs_used": pairs_used,
        "groups_forcing_bits": forced_groups,
        "groups_with_mixed_resolution": mixed_resolution_groups,
        "floor_data_bits": total,
        "trivial_floor_data_bits": len(table.rows) * integer_code_bits(0),
        "statement": (
            "no function whose value is determined by the surviving channels can encode these "
            "observations in fewer than floor_data_bits bits at the declared resolutions"
        ),
    }


# ---------------------------------------------------------------------------
# 7. Adjudication
# ---------------------------------------------------------------------------

ADOPTED = "ADOPTED"
NOT_ADOPTED = "NOT_ADOPTED"
CERTIFIED = "CERTIFIED"
NOT_CERTIFIED = "NOT_CERTIFIED"


def _best(entries: Sequence[CodeLength]) -> CodeLength | None:
    if not entries:
        return None
    return min(entries, key=lambda item: (item.unpriced_bits, item.law_id))


def adjudicate_channel(
    table: ChannelTable,
    laws: Mapping[str, Expr],
    channel: int,
    policy: ChannelPolicy = DECLARED_POLICY,
) -> dict[str, Any]:
    """Is ``channel`` load bearing on ``table``?  Two answers, kept apart on purpose."""

    if not 0 <= channel < table.arity:
        raise ChannelMDLError(f"channel {channel} is outside the table's arity {table.arity}")
    using: list[CodeLength] = []
    ignoring: list[CodeLength] = []
    undecided: list[str] = []
    inadmissible: list[str] = []
    for law_id in sorted(laws):
        expr = laws[law_id]
        for index in referenced_channels(expr):
            if index >= table.arity:
                inadmissible.append(law_id)
                break
        else:
            length = codelength(table, law_id, expr, policy)
            if length is None:
                inadmissible.append(law_id)
                continue
            verdict = channel_dependence(table, expr, channel).verdict
            if verdict == DEPENDS:
                using.append(length)
            elif verdict == INDEPENDENT:
                ignoring.append(length)
            else:
                undecided.append(law_id)
    return _channel_verdict(
        table,
        channel,
        using=using,
        ignoring=ignoring,
        undecided=undecided,
        inadmissible=inadmissible,
        population=len(laws),
        policy=policy,
    )


def _channel_verdict(
    table: ChannelTable,
    channel: int,
    *,
    using: Sequence[CodeLength],
    ignoring: Sequence[CodeLength],
    undecided: Sequence[str],
    inadmissible: Sequence[str],
    population: int,
    policy: ChannelPolicy,
) -> dict[str, Any]:
    """Turn two scored families into the two verdicts.  Shared by laws and by programs."""

    best_using = _best(using)
    best_ignoring = _best(ignoring)

    obstruction = ignoring_codelength_floor(table, (channel,))
    alone = tuple(index for index in range(table.arity) if index != 0)
    obstruction_alone = (
        ignoring_codelength_floor(table, alone) if alone and channel != 0 else None
    )
    exhibited = best_using.data_bits if best_using is not None else None
    certified = (
        best_using is not None
        and exhibited + policy.certification_margin_bits <= obstruction["floor_data_bits"]
    )

    if best_using is None or best_ignoring is None:
        net_bits = None
        adopted = False
    else:
        net_bits = best_ignoring.unpriced_bits - best_using.unpriced_bits
        adopted = net_bits >= policy.channel_admission_bits + policy.adoption_margin_bits

    record: dict[str, Any] = {
        "table_id": table.table_id,
        "channel_index": channel,
        "channel_symbol": channel_symbol(channel),
        "law_space_size": population,
        "laws_using_the_channel": len(using),
        "laws_ignoring_the_channel": len(ignoring),
        "laws_undecided": sorted(undecided),
        "laws_inadmissible": sorted(inadmissible),
        "best_using": best_using.to_dict() if best_using else None,
        "best_ignoring": best_ignoring.to_dict() if best_ignoring else None,
        "obstruction": {
            **obstruction,
            "exhibited_data_bits_using_the_channel": exhibited,
            "gap_bits": (
                obstruction["floor_data_bits"] - exhibited if exhibited is not None else None
            ),
            "verdict": CERTIFIED if certified else NOT_CERTIFIED,
        },
        "adoption": {
            "net_bits": net_bits,
            "admission_bits": policy.channel_admission_bits,
            "adoption_margin_bits": policy.adoption_margin_bits,
            "verdict": ADOPTED if adopted else NOT_ADOPTED,
            "largest_fee_that_still_adopts": (
                net_bits - policy.adoption_margin_bits if net_bits is not None else None
            ),
        },
    }
    alone_certified = (
        obstruction_alone is not None
        and exhibited is not None
        and exhibited + policy.certification_margin_bits
        <= obstruction_alone["floor_data_bits"]
    )
    if obstruction_alone is not None:
        record["obstruction_channel_zero_alone"] = {
            **obstruction_alone,
            "exhibited_data_bits_using_the_channel": exhibited,
            "verdict": CERTIFIED if alone_certified else NOT_CERTIFIED,
        }
    record["headline"] = _headline(
        table,
        channel,
        certified=certified,
        adopted=adopted,
        obstruction=obstruction,
        exhibited=exhibited,
        alone_floor=(
            obstruction_alone["floor_data_bits"] if alone_certified else None
        ),
    )
    return record


def _headline(
    table: ChannelTable,
    channel: int,
    *,
    certified: bool,
    adopted: bool,
    obstruction: Mapping[str, Any],
    exhibited: int | None,
    alone_floor: int | None,
) -> str:
    symbol = channel_symbol(channel)
    others = ", ".join(
        channel_symbol(index) for index in range(table.arity) if index != channel
    )
    if certified and adopted:
        first = channel_symbol(0)
        alone = (
            f"no function of {first} alone codes this table in fewer than {alone_floor} "
            "data bits, and "
            if alone_floor is not None
            else ""
        )
        return (
            f"{alone}no function of {others} alone codes it in fewer than "
            f"{obstruction['floor_data_bits']} data bits; an exhibited law reading channel "
            f"{symbol} codes it in {exhibited}; channel {symbol} is load bearing"
        )
    if certified and not adopted:
        return (
            f"channel {symbol} is forced by the table but the declared price refused it: "
            "the obstruction stands and the adoption verdict does not"
        )
    if adopted:
        return (
            f"channel {symbol} pays for itself inside the declared law space, but no "
            "table-level obstruction rules out a function of the other channels"
        )
    return (
        f"channel {symbol} is not load bearing: it neither clears the declared price nor "
        "forces any bits a function of the other channels could not have saved"
    )


# ---------------------------------------------------------------------------
# 8. Declared tables and the declared law space
# ---------------------------------------------------------------------------

RESOLUTION = Fraction(1, 512)

#: Every declared level is a dyadic rational, so the float a sandboxed program receives is the
#: exact rational the certificate uses and the two lanes cannot silently disagree.
U_LEVELS: tuple[Fraction, ...] = tuple(Fraction(1 + index, 4) for index in range(8))
W_LEVELS: tuple[Fraction, ...] = tuple(Fraction(index, 2) for index in range(5))
X_LEVELS: tuple[Fraction, ...] = tuple(Fraction(index, 4) for index in range(4))
DECLARED_LEVELS = (U_LEVELS, W_LEVELS, X_LEVELS)


# --- The sealed generators.  Nothing on any proposal path reads this block. ------------
#
# They are written in channel indices, so publishing them leaks nothing: the whole point of
# the module is that a channel has no meaning to leak.


def _seal_main(row: Sequence[Fraction]) -> Fraction:
    """Channel 1 is load bearing; channel 2 is a decoy."""

    return row[0] * (2 + row[1])


def _seal_swapped(row: Sequence[Fraction]) -> Fraction:
    """The mirror: channel 2 is load bearing and channel 1 is the decoy."""

    return row[0] * (2 + row[2])


def _seal_weak(row: Sequence[Fraction]) -> Fraction:
    """Channel 2 carries real but barely-resolved signal on top of the main table.

    Its largest reading shifts the observation by exactly half a resolution step, so a law
    ignoring it pays one extra bit on a quarter of the rows and nothing on the rest.  That is
    a real dependence worth far fewer bits than the declared admission fee, which is the case
    the fee exists to refuse.
    """

    return row[0] * (2 + row[1]) + row[2] / 768


def _seal_flat(row: Sequence[Fraction]) -> Fraction:
    """Nothing beyond channel 0 matters."""

    return 3 * row[0]


SEALED_GENERATORS = {
    "main": "obs = u * (2 + w)",
    "swapped": "obs = u * (2 + x)",
    "weak": "obs = u * (2 + w) + x/768",
    "flat": "obs = 3 * u",
    "incomplete": "obs = u * (2 + w), on a design with no witness pair for channel 1",
}


def declared_tables() -> dict[str, ChannelTable]:
    """Four complete designs and one deliberately incomplete one."""

    tables = {
        "main": factorial_table("main", DECLARED_LEVELS, _seal_main, RESOLUTION),
        "swapped": factorial_table("swapped", DECLARED_LEVELS, _seal_swapped, RESOLUTION),
        "weak": factorial_table("weak", DECLARED_LEVELS, _seal_weak, RESOLUTION),
        "flat": factorial_table("flat", DECLARED_LEVELS, _seal_flat, RESOLUTION),
    }
    tables["incomplete"] = incomplete_design()
    return tables


def incomplete_design() -> ChannelTable:
    """One reading of channel 1 per context, so its dependence is undecidable here."""

    rows: list[tuple[Fraction, ...]] = []
    for index, u_value in enumerate(U_LEVELS):
        for x_value in X_LEVELS:
            rows.append((u_value, W_LEVELS[index % len(W_LEVELS)], x_value))
    ordered = tuple(sorted(rows))
    return ChannelTable(
        table_id="incomplete",
        arity=3,
        rows=ordered,
        observations=tuple(_seal_main(row) for row in ordered),
        resolutions=tuple(RESOLUTION for _ in ordered),
    )


def declared_law_space() -> dict[str, Expr]:
    """The declared, finite law space the adoption verdict is relative to."""

    u, w, x = chan(0), chan(1), chan(2)
    return {
        "const_three": const(3),
        "two_u": mul(const(2), u),
        "three_u": mul(const(3), u),
        "five_halves_u": mul(const(Fraction(5, 2)), u),
        "seven_halves_u": mul(const(Fraction(7, 2)), u),
        "four_u": mul(const(4), u),
        "u_squared": mul(u, u),
        "u_plus_w": add(u, w),
        "u_times_w": mul(u, w),
        "u_times_two_plus_w": mul(u, add(const(2), w)),
        "u_times_two_plus_x": mul(u, add(const(2), x)),
        "u_times_two_plus_w_plus_x_over_768": add(
            mul(u, add(const(2), w)), div(x, const(768))
        ),
        "u_times_two_plus_x_plus_w_over_768": add(
            mul(u, add(const(2), x)), div(w, const(768))
        ),
        "three_u_plus_x_over_768": add(mul(const(3), u), div(x, const(768))),
        "u_times_two_plus_w_plus_inert_x": add(mul(u, add(const(2), w)), mul(const(0), x)),
    }


LAW_SPACE_ID = "declared-channel-law-space-1.0"


# ---------------------------------------------------------------------------
# 9. The bridge to executed programs
# ---------------------------------------------------------------------------


def exact_predictions_from_outputs(outputs: Sequence[str]) -> tuple[Fraction, ...]:
    """Sandbox output strings to exact rationals.  The single float-to-exact boundary.

    A sandboxed program is scored by executing it, so its outputs arrive as the decimal
    strings the runner wrote.  Those strings are read as exact decimal rationals here, before
    any arithmetic, and every number downstream of this call is exact.
    """

    values: list[Fraction] = []
    for item in outputs:
        text = item.strip()
        if text.lower() in ("nan", "inf", "-inf", "infinity", "-infinity"):
            raise ChannelMDLError(f"non-finite sandbox output: {item}")
        values.append(Fraction(Decimal(text)))
    return tuple(values)


def program_model_bits(source: str) -> int:
    """The only description length available for arbitrary source: its own bytes.

    Declared, not derived.  Trailing whitespace is stripped per line so the number measures
    the program rather than its formatting.
    """

    normalised = "\n".join(line.rstrip() for line in source.strip().splitlines())
    return 8 * len(normalised.encode("utf-8"))


def codelength_of_program(
    table: ChannelTable,
    law_id: str,
    source: str,
    predicted: Sequence[Fraction],
    policy: ChannelPolicy = DECLARED_POLICY,
) -> CodeLength:
    """Score an executed program on the exact machinery, using its bytes as the model cost."""

    used = program_used_channels(table, predicted)
    return CodeLength(
        law_id=law_id,
        model_bits=program_model_bits(source),
        admission_bits=policy.channel_admission_bits * len(used),
        data_bits=data_bits(residual_codes(table, predicted)),
        used_channels=used,
    )


def program_channel_dependence(
    table: ChannelTable, predicted: Sequence[Fraction], channel: int
) -> Dependence:
    """Behavioural dependence of an executed program, read off its predictions alone.

    Nothing here parses the source.  A program that never mentions the channel and a program
    that mentions it and then cancels it are the same function, and the same answer comes back
    for both -- which is what stops a search from earning credit for decoration.
    """

    if len(predicted) != len(table.rows):
        raise ChannelMDLError("prediction vector does not match the table")
    pairs = table.witness_pairs(channel)
    if not pairs:
        return Dependence(UNDECIDED, 0, None)
    for left, right in pairs:
        if predicted[left] != predicted[right]:
            return Dependence(DEPENDS, len(pairs), (left, right))
    return Dependence(INDEPENDENT, len(pairs), None)


def program_used_channels(
    table: ChannelTable, predicted: Sequence[Fraction]
) -> tuple[int, ...]:
    """The channels an executed program is measured to depend on.  ``UNDECIDED`` counts as used."""

    return tuple(
        index
        for index in range(table.arity)
        if program_channel_dependence(table, predicted, index).verdict in (DEPENDS, UNDECIDED)
    )


def adjudicate_programs(
    table: ChannelTable,
    executed: Mapping[str, tuple[str, Sequence[Fraction]]],
    channel: int,
    policy: ChannelPolicy = DECLARED_POLICY,
) -> dict[str, Any]:
    """The same adjudication, applied to a population the search actually produced.

    ``executed`` maps a program id onto its source and the exact rational predictions its
    sandboxed run produced, one per table row.  The obstruction floor is unchanged -- it is a
    property of the table -- so a population of executed programs is measured against exactly
    the same universally quantified bound as a population of declared expressions.
    """

    if not 0 <= channel < table.arity:
        raise ChannelMDLError(f"channel {channel} is outside the table's arity {table.arity}")
    using: list[CodeLength] = []
    ignoring: list[CodeLength] = []
    undecided: list[str] = []
    for program_id in sorted(executed):
        source, predicted = executed[program_id]
        if len(predicted) != len(table.rows):
            raise ChannelMDLError(f"program {program_id} did not answer every row")
        length = codelength_of_program(table, program_id, source, predicted, policy)
        verdict = program_channel_dependence(table, predicted, channel).verdict
        if verdict == DEPENDS:
            using.append(length)
        elif verdict == INDEPENDENT:
            ignoring.append(length)
        else:
            undecided.append(program_id)
    return _channel_verdict(
        table,
        channel,
        using=using,
        ignoring=ignoring,
        undecided=undecided,
        inadmissible=(),
        population=len(executed),
        policy=policy,
    )


# ---------------------------------------------------------------------------
# 10. Negative controls
# ---------------------------------------------------------------------------


def negative_controls(policy: ChannelPolicy = DECLARED_POLICY) -> dict[str, dict[str, Any]]:
    """Everything that must fail before a positive verdict is allowed to be published."""

    tables = declared_tables()
    laws = declared_law_space()
    main = tables["main"]
    swapped = tables["swapped"]
    weak = tables["weak"]
    flat = tables["flat"]

    decoy = adjudicate_channel(main, laws, 2, policy)
    mirror = adjudicate_channel(swapped, laws, 1, policy)
    weak_priced = adjudicate_channel(weak, laws, 2, policy)
    weak_free = adjudicate_channel(weak, laws, 2, ChannelPolicy(channel_admission_bits=0))
    flat_one = adjudicate_channel(flat, laws, 1, policy)
    flat_two = adjudicate_channel(flat, laws, 2, policy)

    inert = codelength(main, "inert", laws["u_times_two_plus_w_plus_inert_x"], policy)
    plain = codelength(main, "plain", laws["u_times_two_plus_w"], policy)

    narrow = ChannelTable(
        table_id="narrow",
        arity=1,
        rows=tuple((value,) for value in U_LEVELS),
        observations=tuple(3 * value for value in U_LEVELS),
        resolutions=tuple(RESOLUTION for _ in U_LEVELS),
    )
    out_of_range = False
    try:
        codelength(narrow, "reads_channel_one", laws["u_plus_w"], policy)
    except ChannelMDLError:
        out_of_range = True

    tampered = copy.deepcopy(main.to_dict())
    tampered["observations"][0] = _rational_text(main.observations[0] + 1)
    seal_moved = canonical_sha256(tampered) != main.content_sha256()

    undecided = channel_dependence(
        tables["incomplete"], laws["u_times_two_plus_w"], 1
    ).verdict

    return {
        "decoy_channel_on_the_main_table_is_not_adopted": {
            "rejected": decoy["adoption"]["verdict"] == NOT_ADOPTED
            and decoy["obstruction"]["verdict"] == NOT_CERTIFIED,
            "net_bits": decoy["adoption"]["net_bits"],
            "floor_data_bits": decoy["obstruction"]["floor_data_bits"],
        },
        "the_load_bearing_index_is_not_hard_wired": {
            "rejected": mirror["adoption"]["verdict"] == NOT_ADOPTED
            and mirror["obstruction"]["verdict"] == NOT_CERTIFIED,
            "net_bits": mirror["adoption"]["net_bits"],
        },
        "sub_resolution_channel_is_refused_at_the_declared_price": {
            "rejected": weak_priced["adoption"]["verdict"] == NOT_ADOPTED,
            "net_bits": weak_priced["adoption"]["net_bits"],
            "admission_bits": policy.channel_admission_bits,
        },
        "the_price_is_what_refuses_it": {
            "rejected": weak_free["adoption"]["verdict"] == ADOPTED
            and weak_priced["adoption"]["verdict"] == NOT_ADOPTED,
            "largest_fee_that_still_adopts": weak_priced["adoption"][
                "largest_fee_that_still_adopts"
            ],
        },
        "a_table_with_no_extra_dependence_certifies_nothing": {
            "rejected": flat_one["obstruction"]["verdict"] == NOT_CERTIFIED
            and flat_two["obstruction"]["verdict"] == NOT_CERTIFIED
            and flat_one["adoption"]["verdict"] == NOT_ADOPTED
            and flat_two["adoption"]["verdict"] == NOT_ADOPTED,
        },
        "writing_an_inert_channel_term_still_costs_bits": {
            "rejected": inert is not None
            and plain is not None
            and inert.model_bits > plain.model_bits
            and inert.data_bits == plain.data_bits
            and inert.used_channels == plain.used_channels,
        },
        "a_law_reading_a_channel_the_table_lacks_is_refused": {"rejected": out_of_range},
        "an_edited_observation_breaks_the_table_seal": {"rejected": seal_moved},
        "an_incomplete_design_returns_undecided_not_a_guess": {
            "rejected": undecided == UNDECIDED,
            "verdict": undecided,
        },
    }


# ---------------------------------------------------------------------------
# 11. The receipt
# ---------------------------------------------------------------------------

RECEIPT_KEYS = {
    "schema_version",
    "campaign_id",
    "status",
    "claims",
    "scope",
    "policy",
    "policy_sha256",
    "law_space",
    "tables",
    "verdicts",
    "code_audit",
    "widening_audit",
    "blindness",
    "negative_controls",
    "content_sha256",
}

#: Terms that must not reach a channel declaration, a law rendering, or this receipt.  The
#: allowlist on channel symbols is the real guard; this is the belt to its braces.
FORBIDDEN_VOCABULARY = (
    "acceleration",
    "baryonic",
    "cluster",
    "cosmology",
    "curvature",
    "dark",
    "density",
    "distance",
    "galaxy",
    "gravity",
    "halo",
    "luminosity",
    "mass",
    "matter",
    "momentum",
    "orbital",
    "physics",
    "pressure",
    "radius",
    "redshift",
    "rotation",
    "stellar",
    "temperature",
    "velocity",
)


def code_audit() -> dict[str, Any]:
    """Evidence that ``model_bits`` and ``data_bits`` are lengths of messages that decode."""

    tables = declared_tables()
    laws = declared_law_space()
    table = tables["main"]
    rows: list[dict[str, Any]] = []
    for law_id in sorted(laws):
        expr = laws[law_id]
        encoded = encode_expression(expr)
        recovered, position = decode_expression(encoded)
        model_round_trip = position == len(encoded) and recovered == expr
        predicted = predictions(table, expr)
        if predicted is None:
            rows.append(
                {
                    "law_id": law_id,
                    "model_round_trip": model_round_trip,
                    "data_round_trip": None,
                    "worst_reconstruction_error_over_resolution": None,
                }
            )
            continue
        codes = residual_codes(table, predicted)
        message = encode_residuals(codes)
        reconstructed = decode_observations(message, table, predicted)
        worst = max(
            abs(observed - value) / resolution
            for observed, value, resolution in zip(
                table.observations, reconstructed, table.resolutions, strict=True
            )
        )
        rows.append(
            {
                "law_id": law_id,
                "model_round_trip": model_round_trip,
                "model_bits": len(encoded),
                "data_bits": len(message),
                "data_bits_agrees_with_the_scorer": len(message) == data_bits(codes),
                "data_round_trip": worst <= Fraction(1, 2),
                "worst_reconstruction_error_over_resolution": _rational_text(worst),
            }
        )
    return {
        "table_id": table.table_id,
        "every_model_code_decodes": all(item["model_round_trip"] for item in rows),
        "every_data_code_decodes": all(
            item["data_round_trip"] in (True, None) for item in rows
        ),
        "every_reconstruction_is_within_half_a_resolution": all(
            item["data_round_trip"] in (True, None) for item in rows
        ),
        "laws": rows,
    }


def widening_audit(policy: ChannelPolicy = DECLARED_POLICY) -> dict[str, Any]:
    """Adding channels a law does not use must cost that law exactly nothing.

    The wide table and its channel-zero collapse carry the same observations at the same
    resolutions, so a law that reads only channel zero faces the same coding problem on both.
    If widening the slot moved its price, the price would be measuring the declaration rather
    than the law, and every adoption verdict below would be an artefact of the table's shape.
    """

    wide = declared_tables()["main"]
    narrow = collapsed_table(wide, (0,), "main-channel-zero-only")
    laws = declared_law_space()
    rows: list[dict[str, Any]] = []
    for law_id in sorted(laws):
        expr = laws[law_id]
        referenced = referenced_channels(expr)
        if referenced and max(referenced) > 0:
            continue
        wide_length = codelength(wide, law_id, expr, policy)
        narrow_length = codelength(narrow, law_id, expr, policy)
        if wide_length is None or narrow_length is None:
            continue
        rows.append(
            {
                "law_id": law_id,
                "wide_total_bits": wide_length.total_bits,
                "narrow_total_bits": narrow_length.total_bits,
                "identical": wide_length.total_bits == narrow_length.total_bits,
            }
        )
    not_expressible = False
    try:
        codelength(narrow, "reads_channel_one", laws["u_times_two_plus_w"], policy)
    except ChannelMDLError:
        not_expressible = True
    ignoring = codelength(wide, "three_u", laws["three_u"], policy)
    using = codelength(wide, "u_times_two_plus_w", laws["u_times_two_plus_w"], policy)
    return {
        "wide_table_id": wide.table_id,
        "narrow_table_id": narrow.table_id,
        "laws_compared": len(rows),
        "every_channel_ignoring_law_costs_the_same_on_both": bool(rows)
        and all(item["identical"] for item in rows),
        "laws": rows,
        "control_a_channel_using_law_is_not_expressible_on_the_narrow_table": not_expressible,
        "control_a_channel_using_law_does_not_score_the_same_as_a_one_channel_law": (
            ignoring is not None
            and using is not None
            and ignoring.total_bits != using.total_bits
        ),
    }


def blindness_audit() -> dict[str, Any]:
    """No channel carries a name, and no declared string carries a forbidden term."""

    tables = declared_tables()
    laws = declared_law_space()
    surfaces = [LAW_SPACE_ID]
    surfaces.extend(sorted(tables))
    surfaces.extend(sorted(laws))
    surfaces.extend(expression_text(laws[law_id]) for law_id in sorted(laws))
    surfaces.extend(SEALED_GENERATORS.values())
    for table in tables.values():
        surfaces.extend(channel_symbol(index) for index in range(table.arity))
    violations = sorted(
        {
            term
            for term in FORBIDDEN_VOCABULARY
            for surface in surfaces
            if term in surface.lower()
        }
    )
    return {
        "channel_symbols_are_from_the_declared_allowlist": True,
        "declared_surfaces_screened": len(surfaces),
        "forbidden_terms_screened": len(FORBIDDEN_VOCABULARY),
        "violations": violations,
        "note": (
            "channels are addressed by index; the placeholder symbols carry no meaning, so "
            "there is nothing in a declaration for a proposer to read the concept out of"
        ),
    }


def build_receipt(policy: ChannelPolicy = DECLARED_POLICY) -> dict[str, Any]:
    """Adjudicate every declared table and seal the result."""

    tables = declared_tables()
    laws = declared_law_space()
    verdicts: list[dict[str, Any]] = []
    for table_id in sorted(tables):
        table = tables[table_id]
        for channel in range(1, table.arity):
            verdicts.append(adjudicate_channel(table, laws, channel, policy))
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "unlabelled_channels_priced_and_adjudicated_in_exact_integer_arithmetic",
        "claims": dict(CLAIMS),
        "scope": SCOPE,
        "policy": policy.to_dict(),
        "policy_sha256": policy.content_sha256(),
        "law_space": {
            "law_space_id": LAW_SPACE_ID,
            "size": len(laws),
            "laws": [
                {
                    "law_id": law_id,
                    "expression_text": expression_text(laws[law_id]),
                    "model_bits": model_bits(laws[law_id]),
                    "referenced_channels": list(referenced_channels(laws[law_id])),
                }
                for law_id in sorted(laws)
            ],
            "note": (
                "the adoption verdict is relative to this finite space; the obstruction "
                "verdict is not relative to anything"
            ),
        },
        "tables": [
            {
                "table_id": table_id,
                "arity": tables[table_id].arity,
                "rows": len(tables[table_id].rows),
                "complete_factorial_design": bool(tables[table_id].witness_pairs(1)),
                "resolution": _rational_text(tables[table_id].resolutions[0]),
                "sealed_generator": SEALED_GENERATORS[table_id],
                "content_sha256": tables[table_id].content_sha256(),
            }
            for table_id in sorted(tables)
        ],
        "verdicts": verdicts,
        "code_audit": code_audit(),
        "widening_audit": widening_audit(policy),
        "blindness": blindness_audit(),
        "negative_controls": negative_controls(policy),
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(value: Mapping[str, Any]) -> None:
    """The receipt boundary: no adoption without a price, no certificate without a floor."""

    if set(value) != RECEIPT_KEYS:
        raise ChannelMDLError("channel-MDL receipt boundary changed")
    if value["schema_version"] != RECEIPT_SCHEMA or value["campaign_id"] != CAMPAIGN_ID:
        raise ChannelMDLError("channel-MDL receipt identity changed")
    if value["claims"] != CLAIMS:
        raise ChannelMDLError("channel-MDL claims block changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value["content_sha256"] != canonical_sha256(body):
        raise ChannelMDLError("channel-MDL receipt seal changed")
    if value["policy_sha256"] != canonical_sha256(value["policy"]):
        raise ChannelMDLError("channel-MDL policy binding changed")
    if value["policy"]["channel_admission_bits"] <= 0:
        raise ChannelMDLError("a receipt may not publish a free channel")
    if value["blindness"]["violations"]:
        raise ChannelMDLError("a declared surface leaked a forbidden term")
    if not value["code_audit"]["every_model_code_decodes"]:
        raise ChannelMDLError("a model codelength does not correspond to a decodable message")
    if not value["code_audit"]["every_data_code_decodes"]:
        raise ChannelMDLError("a data codelength does not correspond to a decodable message")
    widening = value["widening_audit"]
    if not widening["every_channel_ignoring_law_costs_the_same_on_both"]:
        raise ChannelMDLError("widening the slot moved the price of a law that ignores it")
    if not widening["control_a_channel_using_law_is_not_expressible_on_the_narrow_table"]:
        raise ChannelMDLError("the widening control did not fail as it must")
    if not widening[
        "control_a_channel_using_law_does_not_score_the_same_as_a_one_channel_law"
    ]:
        raise ChannelMDLError("the widening control did not fail as it must")
    controls = value["negative_controls"]
    if set(controls) != set(negative_controls(ChannelPolicy(**_policy_fields(value["policy"])))):
        raise ChannelMDLError("channel-MDL negative controls changed")
    if not all(control["rejected"] is True for control in controls.values()):
        raise ChannelMDLError("a channel-MDL negative control passed")
    seen = set()
    for verdict in value["verdicts"]:
        key = (verdict["table_id"], verdict["channel_index"])
        if key in seen:
            raise ChannelMDLError(f"duplicate verdict for {key}")
        seen.add(key)
        adoption = verdict["adoption"]
        obstruction = verdict["obstruction"]
        if adoption["verdict"] == ADOPTED:
            if adoption["net_bits"] is None:
                raise ChannelMDLError("an adopted channel carries no net bit count")
            if adoption["net_bits"] < (
                adoption["admission_bits"] + adoption["adoption_margin_bits"]
            ):
                raise ChannelMDLError("an adopted channel did not repay its declared price")
        if obstruction["verdict"] == CERTIFIED:
            exhibited = obstruction["exhibited_data_bits_using_the_channel"]
            if exhibited is None:
                raise ChannelMDLError("a certified channel exhibits no law")
            if exhibited >= obstruction["floor_data_bits"]:
                raise ChannelMDLError("a certificate was issued without clearing the floor")
            if obstruction["floor_data_bits"] <= obstruction["trivial_floor_data_bits"]:
                raise ChannelMDLError("a certificate rests on the trivial floor")
        if not verdict["headline"]:
            raise ChannelMDLError("a verdict carries no plain-language statement")


def _policy_fields(value: Mapping[str, Any]) -> dict[str, int]:
    return {
        "channel_admission_bits": int(value["channel_admission_bits"]),
        "adoption_margin_bits": int(value["adoption_margin_bits"]),
        "certification_margin_bits": int(value["certification_margin_bits"]),
    }


# ---------------------------------------------------------------------------
# 12. CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigma-unlabelled-channel-mdl")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build_receipt()
    validate_receipt(result)
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
