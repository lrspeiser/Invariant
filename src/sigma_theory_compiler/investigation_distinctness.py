"""The distinctness gate: N investigations that DO different things, proven by running them.

Ask a model for twenty investigations and you reliably get twenty phrasings of one.  The
measured instance is on record in :mod:`.creativity_measure`: a live run returned six
syntactically distinct programs -- ``u/(1 - u/2 + u*u/6)``, ``u*(1 + u)**-0.25``,
``2*u/(2 + u)`` and others -- every one of which is the identity map over the declared domain.
Source diversity was six.  Behavioural diversity was one.  Any gate that reads source text
would have passed all six.

:mod:`.creativity_measure` solved that for PROGRAMS by clustering output vectors.  This module
generalises the same principle to INVESTIGATIONS -- proposals that may be generators, problems
or constraints rather than functions -- and turns the measurement into a GATE: a batch goes in,
the behaviourally distinct ones come out, and the count that was rejected comes out beside them.

WHAT A FINGERPRINT IS, PER PROPOSAL KIND
========================================

A fingerprint is always computed by EXECUTING the proposal against a declared probe.  Nothing is
ever read off the source text, because the source text is exactly what a rephrasing changes.

``generator`` -- what does it PRODUCE?
    Entry ``generate(index) -> row of numbers``, called for ``index = 0 .. n-1``.  The
    fingerprint is the SORTED tuple of quantised rows.  Sorted deliberately: a generator that
    emits the same basis in a different order is the same generator, so two generators differ
    only when the SET they emit differs.

``problem`` -- what does it RANK?
    Entry ``score(*answer) -> number``, called once per rung of a declared reference ladder of
    answers.  The fingerprint is the ORDERED tuple of quantised scores.  Ordered deliberately:
    rung i's score belongs to rung i.  A problem IS the ordering it imposes on answers, and two
    problems are the same problem exactly when no answer in the ladder can tell them apart.

``constraint`` -- what does it ADMIT?
    Entry ``admits(*witness) -> number``, called once per witness in a declared panel, admitting
    when the returned value is nonzero.  The fingerprint is the ORDERED tuple of admit bits.
    Reduced to a bit deliberately: a constraint returning 0.7 where another returns 1.0 has cut
    the panel along the same line, and where the line falls is the whole of what a constraint is.

``program`` -- what does it COMPUTE?
    Entry ``rule(*point) -> number`` over declared probe points; the ORDERED tuple of quantised
    outputs.  This is :mod:`.creativity_measure`'s notion, kept intact so this module is a
    superset of it rather than a competing second opinion.

Three rejection verdicts, and only three: ``duplicate_behaviour`` (its fingerprint is one
already admitted from this batch), ``duplicate_of_incumbent`` (its fingerprint is one the system
already has, which is what stops a "third generator" that behaves like an existing one), and
``unrunnable`` (the sandbox returned a typed failure).  An unrunnable proposal is REJECTED, never
counted as a distinct behaviour -- two proposals that both fail to parse are not two ways of
investigating anything, and a gate that clustered failures would report a syntax error as
diversity.

RESOLUTION IS DECLARED, NEVER HIDDEN
====================================

Two behaviours are the same when their quantised coordinates are equal, and the quantisation --
``significant_digits`` on the probe -- is part of the probe's declaration and its digest.  This
is load-bearing and is not a tuning knob to be minimised.  The documented six programs agree to
about 5e-7 relative, so they collapse to one behaviour at the default six significant digits and
SEPARATE at nine.  Both answers are correct answers to different questions, and the gate refuses
to pretend there is a resolution-free fact of the matter.  ``coarse_collision_pairs`` reports how
many admitted pairs would have merged at one digit coarser, so a split that hangs on the declared
resolution is visible in the receipt rather than silently decisive.

MODEL OUTPUT IS A PROPOSAL, NEVER A RESULT
==========================================

Every proposal carries model-authored text: its ``source`` and whatever it ``claims`` about
itself.  None of it reaches the receipt.  The receipt has two zones and the split is enforced,
not merely intended:

* ``verified`` -- every leaf was produced by this module RUNNING the proposal: fingerprints,
  cluster occupancies, counts, typed sandbox reasons.
* ``unverified_model_claims`` -- the model's own text, present only as SHA-256 digests under an
  explicit ``unverified_proposal`` status.  Never verbatim, never as a finding.

:func:`assert_receipt_is_model_free` walks the sealed receipt and raises :class:`ModelValueLeak`
if any model-authored scalar appears in it, and :func:`run_distinctness_gate` calls it before
sealing.  A future edit that copies a claim into the receipt therefore fails at runtime rather
than shipping a laundered assertion.

The one thing that is NOT a leak: a value the proposal's own code computed and the sandbox
returned.  That value passed through execution, which is the exact verification the rule demands.
A model may put any number it likes into the receipt by writing a program that computes it -- and
that is the whole point, because then the number is a measured output, not an assertion.

The check errs toward refusing.  If a model happens to claim a string equal to something the gate
computed -- a ratio, a digest -- the batch is refused rather than sealed, because the gate cannot
tell a coincidence from a laundered claim and the invariant outranks throughput.  That direction
is deliberate and is pinned by a test.

Claim boundary: distinctness here means "these proposals produced different measured behaviour on
this declared probe".  It never means a proposal is correct, useful, novel with respect to the
literature, or worth running.  A batch can be perfectly distinct and uniformly worthless, and the
gate will say so by admitting all of it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .funsearch_loop import SANDBOX_FAILURE_REASONS, SandboxBudget, run_in_sandbox
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-investigation-distinctness-gate-1.0"

#: The proposal kinds this gate knows how to run.  A kind outside this tuple is refused rather
#: than served by whichever reduction happens to be wired in -- the same rule
#: :data:`.tensor_constraint_search.SUPPORTED_GENERATORS` applies to declared frameworks.
PROPOSAL_KINDS: tuple[str, ...] = ("constraint", "generator", "problem", "program")

#: Six significant digits.  Chosen against the measured failure rather than tuned: the six
#: documented live-run programs agree to about 5e-7 relative, so six digits calls them one
#: behaviour and nine calls them six.  Callers whose question has a finer resolution must declare
#: it on the probe, and the receipt will carry the number they declared.
DEFAULT_SIGNIFICANT_DIGITS = 6

#: Written into every receipt so a reader never has to reconstruct what was compared.
FINGERPRINT_RULES: dict[str, dict[str, str]] = {
    "constraint": {
        "entry_signature": "admits(*witness) -> number, nonzero admits",
        "probe_is": "a declared witness panel",
        "reduction": "ordered tuple of admit bits, one per witness",
        "ordering": "ordered -- witness i's bit belongs to witness i",
        "identity": "two constraints are the same when they cut the panel along the same line",
    },
    "generator": {
        "entry_signature": "generate(index) -> row of numbers",
        "probe_is": "index = 0 .. count-1",
        "reduction": "sorted tuple of quantised rows",
        "ordering": "sorted -- emission order is not part of a generator's identity",
        "identity": "two generators are the same when the set of rows they emit is the same",
    },
    "problem": {
        "entry_signature": "score(*answer) -> number",
        "probe_is": "a declared reference ladder of answers",
        "reduction": "ordered tuple of quantised scores, one per rung",
        "ordering": "ordered -- rung i's score belongs to rung i",
        "identity": "two problems are the same when no answer in the ladder separates them",
    },
    "program": {
        "entry_signature": "rule(*point) -> number",
        "probe_is": "declared probe points",
        "reduction": "ordered tuple of quantised outputs",
        "ordering": "ordered -- point i's output belongs to point i",
        "identity": "creativity_measure's notion, preserved so this gate is a superset of it",
    },
}

#: Rejection verdicts.  Exhaustive by construction: every proposal receives exactly one of these
#: or is admitted.
REJECTION_VERDICTS: tuple[str, ...] = (
    "duplicate_behaviour",
    "duplicate_of_incumbent",
    "unrunnable",
)

#: Leaves too information-free to be a smuggled finding.  A model asserting ``true`` or ``0`` has
#: asserted nothing a reader could mistake for a measured result; the leak check targets the
#: values that could be -- digests, verdicts, dimensions, scores, counts.
UNINFORMATIVE_LEAVES: frozenset[str] = frozenset(
    {"null", "bool:true", "bool:false", "int:0", "int:1", "int:-1", "int:2", "str:"}
)

#: The COMPLETE set of non-measured strings the gate itself writes into the verified zone.  It is
#: closed on purpose and :func:`assert_receipt_is_model_free` enforces the closure: a verified
#: string leaf must be one of these words, a proposal id, a 64-hex digest or a decimal string, and
#: anything else raises.  That is what lets the leak check ignore a model claim that happens to
#: collide with one of these words -- a model claiming ``"kind": "program"`` cannot thereby put
#: anything into a receipt, because the receipt's ``kind`` came from the probe either way.
GATE_VOCABULARY: frozenset[str] = frozenset(
    PROPOSAL_KINDS + REJECTION_VERDICTS + SANDBOX_FAILURE_REASONS + ("unverified_proposal",)
)

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_ENTRY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


class DistinctnessError(ValueError):
    """A caller error: a malformed probe, a malformed proposal, or a kind mismatch.

    Distinct from a model error.  A model that writes an unrunnable program is an expected,
    typed, logged event; a caller that hands a ``generator`` proposal to a ``problem`` probe has
    made a mistake the gate must not paper over by guessing which one was meant.
    """


class ModelValueLeak(ValueError):
    """A model-authored value reached the verified zone of a receipt.

    This is the failure the whole module is built to make impossible: a proposal's own assertion
    -- a claimed dimension, a claimed score, a claimed verdict -- surviving into a sealed
    artifact where a later reader would take it for something the machine measured.
    """


# ---------------------------------------------------------------------------
# 1. Declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Proposal:
    """One model-authored investigation.  Untrusted in full: ``source`` and ``claims`` alike.

    ``proposal_id`` is a caller-assigned slug, validated to a narrow character class so it can
    label a row in a receipt without becoming a channel for model prose.  It is an address, not
    a finding, and it is the only proposal field that appears in a receipt unhashed.
    """

    proposal_id: str
    kind: str
    entry: str
    source: str
    claims: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _IDENTIFIER.match(self.proposal_id or ""):
            raise DistinctnessError(
                f"proposal_id {self.proposal_id!r} is not a lowercase slug of at most 64 "
                "characters; ids label receipt rows and are deliberately not free text"
            )
        if self.kind not in PROPOSAL_KINDS:
            raise DistinctnessError(
                f"proposal kind {self.kind!r} is not one this gate implements "
                f"{list(PROPOSAL_KINDS)}; refusing to fingerprint it with another kind's rule"
            )
        if not _ENTRY.match(self.entry or ""):
            raise DistinctnessError(f"entry {self.entry!r} is not a Python identifier")
        if not isinstance(self.source, str):
            raise DistinctnessError("proposal source must be text")
        if not isinstance(self.claims, Mapping):
            raise DistinctnessError("proposal claims must be a mapping")

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    @property
    def claims_sha256(self) -> str:
        return canonical_sha256({"claims": _stringify(self.claims)})


@dataclass(frozen=True, slots=True)
class ProbeSuite:
    """What the proposals are run against, and at what resolution behaviours count as equal.

    The probe carries the resolution because resolution is not a global constant: what counts as
    "the same behaviour" is a property of the question being asked, and the question lives here.
    """

    kind: str
    inputs: tuple[tuple[float, ...], ...]
    output_width: int = 1
    significant_digits: int = DEFAULT_SIGNIFICANT_DIGITS
    label: str = "unlabelled_probe"

    def __post_init__(self) -> None:
        if self.kind not in PROPOSAL_KINDS:
            raise DistinctnessError(f"probe kind {self.kind!r} is not in {list(PROPOSAL_KINDS)}")
        if not self.inputs:
            raise DistinctnessError(
                "a probe with no inputs cannot distinguish anything: every proposal would "
                "fingerprint to the empty tuple and the batch would collapse to one"
            )
        if int(self.output_width) < 1:
            raise DistinctnessError("output_width must be at least 1")
        if self.kind == "constraint" and int(self.output_width) != 1:
            raise DistinctnessError(
                "a constraint returns one number per witness and admits when it is nonzero; "
                "a wider return has no declared reduction to an admit bit"
            )
        if not 1 <= int(self.significant_digits) <= 15:
            raise DistinctnessError(
                "significant_digits outside 1..15; below 1 nothing is distinguishable and above "
                "15 the comparison is reading double-precision replay noise"
            )
        widths = {len(row) for row in self.inputs}
        if len(widths) != 1:
            raise DistinctnessError(f"probe input rows have mixed arity {sorted(widths)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "point_count": len(self.inputs),
            "input_arity": len(self.inputs[0]),
            "output_width": int(self.output_width),
            "significant_digits": int(self.significant_digits),
            # Decimal strings, never floats: canonical_sha256 forbids floats so a receipt can
            # never carry cross-runtime serialization drift.
            "inputs": [[format(float(value), ".17g") for value in row] for row in self.inputs],
        }

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def program_probe(
    points: Sequence[float],
    *,
    significant_digits: int = DEFAULT_SIGNIFICANT_DIGITS,
    label: str = "program_probe",
) -> ProbeSuite:
    """``rule(point)`` over one-dimensional probe points."""

    return ProbeSuite(
        "program",
        tuple((float(point),) for point in points),
        1,
        significant_digits,
        label,
    )


def generator_probe(
    count: int,
    width: int,
    *,
    significant_digits: int = DEFAULT_SIGNIFICANT_DIGITS,
    label: str = "generator_probe",
) -> ProbeSuite:
    """``generate(index)`` for ``index = 0 .. count-1``, each returning ``width`` numbers."""

    return ProbeSuite(
        "generator",
        tuple((float(index),) for index in range(int(count))),
        int(width),
        significant_digits,
        label,
    )


def problem_probe(
    ladder: Sequence[Sequence[float]],
    *,
    significant_digits: int = DEFAULT_SIGNIFICANT_DIGITS,
    label: str = "problem_probe",
) -> ProbeSuite:
    """``score(*answer)`` over a declared reference ladder of candidate answers."""

    return ProbeSuite(
        "problem",
        tuple(tuple(float(value) for value in rung) for rung in ladder),
        1,
        significant_digits,
        label,
    )


def constraint_probe(
    panel: Sequence[Sequence[float]],
    *,
    label: str = "constraint_probe",
) -> ProbeSuite:
    """``admits(*witness)`` over a declared witness panel.  Resolution is irrelevant: bits."""

    return ProbeSuite(
        "constraint",
        tuple(tuple(float(value) for value in witness) for witness in panel),
        1,
        1,
        label,
    )


# ---------------------------------------------------------------------------
# 2. Fingerprints, computed by running
# ---------------------------------------------------------------------------


def quantise(text: str, significant_digits: int) -> str:
    """One measured number as a normalised decimal string at the declared resolution.

    ``-0.0`` is folded onto ``0.0`` first: a proposal that reaches zero from below has not found
    a different behaviour from one that reaches it from above, and letting the sign of a zero
    split a cluster would be reading a floating-point artifact as diversity.
    """

    number = float(text)
    if not math.isfinite(number):
        raise DistinctnessError(f"non-finite sandbox output {text!r} reached quantisation")
    if number == 0.0:
        number = 0.0
    return format(number, f".{int(significant_digits) - 1}e")


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """What one proposal DID, plus the digest that is its behavioural identity."""

    proposal_id: str
    kind: str
    ok: bool
    reason: str
    detail: str
    coordinates: tuple[str, ...]
    raw: tuple[str, ...]
    digest: str

    def to_row(self) -> dict[str, Any]:
        """The receipt row.  Every value here was produced by running the proposal."""

        return {
            "proposal_id": self.proposal_id,
            "kind": self.kind,
            "ran": self.ok,
            "fingerprint_sha256": self.digest,
            "coordinate_count": len(self.coordinates),
        }


def _reduce(
    outputs: Sequence[str], probe: ProbeSuite
) -> tuple[str, ...]:
    """Fold a flat sandbox output list into the kind's declared fingerprint coordinates."""

    width = int(probe.output_width)
    if len(outputs) != len(probe.inputs) * width:
        raise DistinctnessError(
            f"sandbox returned {len(outputs)} values for {len(probe.inputs)} inputs at width "
            f"{width}; the sandbox contract was violated"
        )
    rows = [tuple(outputs[i : i + width]) for i in range(0, len(outputs), width)]
    digits = int(probe.significant_digits)
    if probe.kind == "constraint":
        return tuple("1" if float(row[0]) != 0.0 else "0" for row in rows)
    quantised = [tuple(quantise(value, digits) for value in row) for row in rows]
    if probe.kind == "generator":
        quantised = sorted(quantised)
    return tuple(value for row in quantised for value in row)


def _digest_of(kind: str, probe_sha256: str, coordinates: Sequence[str]) -> str:
    """The behavioural identity: kind, probe and coordinates, sealed together.

    The probe digest is inside the fingerprint on purpose.  A fingerprint is a statement about
    behaviour ON A DECLARED PROBE, and two fingerprints taken against different probes are not
    comparable; binding the probe in makes the incomparability mechanical instead of a footnote.
    """

    return canonical_sha256(
        {
            "schema_version": SCHEMA,
            "kind": kind,
            "probe_sha256": probe_sha256,
            "coordinates": list(coordinates),
        }
    )


def fingerprint_of(
    proposal: Proposal,
    probe: ProbeSuite,
    *,
    budget: SandboxBudget | None = None,
) -> Fingerprint:
    """Run one proposal and return what it DID.  Never reads the source except to run it."""

    if proposal.kind != probe.kind:
        raise DistinctnessError(
            f"proposal {proposal.proposal_id!r} declares kind {proposal.kind!r} but the probe is "
            f"{probe.kind!r}; the two reductions measure different things and guessing which was "
            "meant would produce a fingerprint nobody declared"
        )
    envelope = budget or SandboxBudget()
    outcome = run_in_sandbox(
        proposal.source,
        proposal.entry,
        probe.inputs,
        envelope,
        output_width=int(probe.output_width),
    )
    probe_sha256 = probe.content_sha256
    if not outcome.ok:
        reason = outcome.reason if outcome.reason in SANDBOX_FAILURE_REASONS else "child_crashed"
        return Fingerprint(
            proposal.proposal_id,
            proposal.kind,
            False,
            reason,
            outcome.detail[:200],
            (),
            (),
            # A failure gets a digest that cannot collide with any behaviour and cannot collide
            # with another failure: unrunnable proposals must never cluster into "one distinct
            # way of failing" and be counted as diversity.
            canonical_sha256(
                {
                    "schema_version": SCHEMA,
                    "unrunnable": True,
                    "proposal_id": proposal.proposal_id,
                    "reason": reason,
                }
            ),
        )
    coordinates = _reduce(outcome.outputs, probe)
    return Fingerprint(
        proposal.proposal_id,
        proposal.kind,
        True,
        "",
        "",
        coordinates,
        tuple(outcome.outputs),
        _digest_of(proposal.kind, probe_sha256, coordinates),
    )


# ---------------------------------------------------------------------------
# 3. The model-value leak check
# ---------------------------------------------------------------------------


def _stringify(value: Any) -> Any:
    """A model claim rendered into the canonical-JSON subset, floats included, as text."""

    if isinstance(value, Mapping):
        return {str(key): _stringify(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stringify(child) for child in value]
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, (bool, int)) or value is None:
        return value
    return str(value)


def _render(leaf: Any) -> str:
    """A typed rendering, so ``True``, ``1`` and ``"1"`` are three different leaves."""

    if leaf is None:
        return "null"
    if isinstance(leaf, bool):
        return "bool:true" if leaf else "bool:false"
    if isinstance(leaf, int):
        return f"int:{leaf}"
    if isinstance(leaf, float):
        return f"float:{leaf!r}"
    return f"str:{leaf}"


def _walk(value: Any, path: str = "$") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")
    else:
        yield path, value


def model_authored_leaves(proposals: Sequence[Proposal]) -> frozenset[str]:
    """Every scalar the model wrote, typed-rendered, minus the ones that carry no information."""

    leaves: set[str] = set()
    for proposal in proposals:
        leaves.add(_render(proposal.source))
        for _, leaf in _walk(dict(proposal.claims)):
            leaves.add(_render(leaf))
    ignorable = UNINFORMATIVE_LEAVES | {f"str:{word}" for word in GATE_VOCABULARY}
    return frozenset(leaves - ignorable)


def assert_receipt_is_model_free(
    receipt: Mapping[str, Any], proposals: Sequence[Proposal]
) -> None:
    """Raise if a model-authored value reached the receipt, or if the verified zone grew free text.

    Two checks, because either alone has a hole.  The first scans the whole receipt for a value
    the model wrote.  The second pins the verified zone's string vocabulary CLOSED -- every string
    leaf must be a declared gate word, a proposal id, a 64-hex digest or a decimal string -- so a
    later edit cannot open a free-text field through which a claim could travel under a name the
    first check was told to ignore.

    Called by :func:`run_distinctness_gate` before the receipt is sealed, so the invariant holds
    structurally: an edit that copies a claim through fails here rather than shipping.
    """

    identifiers = {proposal.proposal_id for proposal in proposals}
    forbidden = model_authored_leaves(proposals)
    for path, leaf in _walk(receipt):
        rendered = _render(leaf)
        if rendered in forbidden:
            raise ModelValueLeak(
                f"model-authored value reached the receipt at {path}: {rendered[:120]!r}.  "
                "Model output is a proposal, never a result: it may enter a receipt only as a "
                "digest under unverified_model_claims, or as a number the sandbox measured by "
                "running the proposal's own code."
            )
    for path, leaf in _walk(receipt.get("verified", {})):
        if not isinstance(leaf, str) or not leaf:
            continue
        if leaf in GATE_VOCABULARY or leaf in identifiers:
            continue
        if _DIGEST.match(leaf) or _DECIMAL.match(leaf):
            continue
        raise ModelValueLeak(
            f"undeclared free text in the verified zone at {path}: {leaf[:120]!r}.  The verified "
            "zone carries only declared gate words, proposal ids, digests and decimal strings, "
            "because a free-text field there is a channel a model claim could travel through."
        )


# ---------------------------------------------------------------------------
# 4. The gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateResult:
    """What survived, what it measured, and the sealed receipt that says so.

    ``admitted`` carries live :class:`Proposal` objects for the caller to use.  The receipt
    carries none of their text -- only digests -- which is why the two are separate fields
    rather than one nested structure.
    """

    admitted: tuple[Proposal, ...]
    fingerprints: tuple[Fingerprint, ...]
    receipt: dict[str, Any]

    @property
    def proposals_generated(self) -> int:
        return int(self.receipt["verified"]["proposals_generated"])

    @property
    def distinct_proposals(self) -> int:
        return int(self.receipt["verified"]["distinct_proposals"])

    @property
    def rejected(self) -> int:
        return int(self.receipt["verified"]["rejected"])


def _effective_number(sizes: Sequence[int]) -> float:
    """``exp(H)`` over cluster occupancies -- creativity_measure's rule, same reason.

    A raw count says 21 when a batch is one investigation repeated twenty times plus one other.
    The effective number says about 2, which is what a reader means by "how many different
    things were actually tried".
    """

    total = sum(sizes)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for size in sizes:
        if size <= 0:
            continue
        share = size / total
        entropy -= share * math.log(share)
    return math.exp(entropy)


def _coarse_collisions(
    admitted: Sequence[Fingerprint], probe: ProbeSuite
) -> int:
    """Admitted pairs that would have merged at one digit coarser.

    A nonzero count does not mean the gate was wrong.  It means the split between those pairs is
    a function of the declared resolution, and a reader who cares should say so out loud rather
    than discover it later.
    """

    digits = int(probe.significant_digits)
    if digits <= 1 or probe.kind == "constraint":
        return 0
    coarse: list[tuple[str, ...]] = []
    coarser = ProbeSuite(
        probe.kind, probe.inputs, probe.output_width, digits - 1, probe.label
    )
    for item in admitted:
        if not item.ok:
            continue
        coarse.append(_reduce(item.raw, coarser))
    pairs = 0
    for i in range(len(coarse)):
        for j in range(i + 1, len(coarse)):
            if coarse[i] == coarse[j]:
                pairs += 1
    return pairs


def run_distinctness_gate(
    proposals: Sequence[Proposal],
    probe: ProbeSuite,
    *,
    incumbents: Sequence[Proposal] = (),
    budget: SandboxBudget | None = None,
) -> GateResult:
    """Fingerprint a batch by running it, keep the behaviourally distinct, count the rest.

    ``incumbents`` are the behaviours the system already has.  Running them through the same
    probe is what makes "a third generator" mean a third BEHAVIOUR: a proposal that reproduces
    an existing generator is rejected as ``duplicate_of_incumbent``, however new its source is.
    An incumbent that fails to run is a caller error and stops the gate, because incumbents are
    the system's own code and a comparison against code that will not run is meaningless.
    """

    envelope = budget or SandboxBudget()
    seen: dict[str, str] = {}
    incumbent_rows: list[dict[str, Any]] = []
    for item in incumbents:
        mark = fingerprint_of(item, probe, budget=envelope)
        if not mark.ok:
            raise DistinctnessError(
                f"incumbent {item.proposal_id!r} did not run ({mark.reason}: {mark.detail}); "
                "an incumbent is the system's own declared behaviour, so a failure here is a "
                "broken baseline rather than a rejected proposal"
            )
        seen.setdefault(mark.digest, f"incumbent:{item.proposal_id}")
        incumbent_rows.append(mark.to_row())

    fingerprints: list[Fingerprint] = []
    admitted: list[Proposal] = []
    admitted_marks: list[Fingerprint] = []
    cluster_members: dict[str, list[str]] = {}
    rejected_rows: list[dict[str, Any]] = []
    verdict_counts = dict.fromkeys(REJECTION_VERDICTS, 0)
    sandbox_reasons: dict[str, int] = {}

    for proposal in proposals:
        mark = fingerprint_of(proposal, probe, budget=envelope)
        fingerprints.append(mark)
        if not mark.ok:
            verdict_counts["unrunnable"] += 1
            sandbox_reasons[mark.reason] = sandbox_reasons.get(mark.reason, 0) + 1
            rejected_rows.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "verdict": "unrunnable",
                    "sandbox_reason": mark.reason,
                    "duplicate_of": None,
                }
            )
            continue
        owner = seen.get(mark.digest)
        if owner is None:
            seen[mark.digest] = proposal.proposal_id
            cluster_members[mark.digest] = [proposal.proposal_id]
            admitted.append(proposal)
            admitted_marks.append(mark)
            continue
        verdict = (
            "duplicate_of_incumbent"
            if owner.startswith("incumbent:")
            else "duplicate_behaviour"
        )
        verdict_counts[verdict] += 1
        if verdict == "duplicate_behaviour":
            cluster_members[mark.digest].append(proposal.proposal_id)
        rejected_rows.append(
            {
                "proposal_id": proposal.proposal_id,
                "verdict": verdict,
                "sandbox_reason": None,
                "duplicate_of": owner.split(":", 1)[-1],
            }
        )

    runnable = sum(1 for mark in fingerprints if mark.ok)
    distinct = len(admitted)
    sizes = [len(cluster_members[mark.digest]) for mark in admitted_marks]
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "declared": {
            "probe": probe.to_dict(),
            "probe_sha256": probe.content_sha256,
            "fingerprint_rule": FINGERPRINT_RULES[probe.kind],
            "sandbox": envelope.to_dict(),
            "rejection_verdicts": list(REJECTION_VERDICTS),
            "policy": (
                "a fingerprint is computed by running the proposal against the declared probe; "
                "two proposals are the same investigation when their fingerprints are equal at "
                "the declared resolution, however differently they are written"
            ),
        },
        "verified": {
            "proposals_generated": len(proposals),
            "runnable_proposals": runnable,
            "distinct_proposals": distinct,
            "rejected": len(rejected_rows),
            "rejected_by_verdict": dict(verdict_counts),
            "unrunnable_by_sandbox_reason": dict(sorted(sandbox_reasons.items())),
            "admitted": [
                dict(mark.to_row(), cluster_size=len(cluster_members[mark.digest]))
                for mark in admitted_marks
            ],
            "rejected_detail": rejected_rows,
            "incumbents": incumbent_rows,
            # Decimal strings: canonical_sha256 forbids floats.
            "effective_distinct_investigations": format(_effective_number(sizes), ".6f"),
            "wasted_variation_ratio": format(
                runnable / distinct if distinct else 0.0, ".6f"
            ),
            "coarse_collision_pairs": _coarse_collisions(admitted_marks, probe),
        },
        "unverified_model_claims": {
            proposal.proposal_id: {
                "status": "unverified_proposal",
                "source_sha256": proposal.source_sha256,
                "claims_sha256": proposal.claims_sha256,
            }
            for proposal in proposals
        },
        "claims": {
            "fingerprint_computed_by_running_not_by_reading": True,
            "model_output_is_a_proposal_never_a_result": True,
            "unrunnable_proposals_are_rejected_not_counted_as_distinct": True,
            "distinctness_is_behavioural_not_textual": True,
            "distinctness_asserts_difference_never_correctness": True,
        },
        "notes": [
            (
                "wasted_variation_ratio is runnable proposals per distinct behaviour; 1.0 means "
                "every proposal did something new and a large value is the interesting number"
            ),
            (
                "coarse_collision_pairs counts admitted pairs that would merge at one digit "
                "coarser, so a resolution-dependent split is visible rather than silent"
            ),
        ],
    }
    # Structural, not aspirational: the receipt is checked before it is sealed, so a future edit
    # that copies a model claim into it fails here.
    assert_receipt_is_model_free(receipt, list(proposals) + list(incumbents))
    receipt["content_sha256"] = canonical_sha256(receipt)
    return GateResult(tuple(admitted), tuple(fingerprints), receipt)


def proposals_from_sources(
    sources: Sequence[str],
    kind: str,
    entry: str,
    *,
    prefix: str = "proposed",
) -> tuple[Proposal, ...]:
    """Adapt raw proposer output into proposals whose ids the GATE assigned, not the model.

    The bridge every caller needs, and the place one rule is enforced: a proposal's id comes from
    its position in the batch.  Receipt rows are addressed by id, so an id the model chose would
    be a channel through which model prose could reach a receipt under a structural name.  A
    proposer returns source code and nothing else; everything that labels it is assigned here.
    """

    return tuple(
        Proposal(f"{prefix}_{index:03d}", kind, entry, str(source))
        for index, source in enumerate(sources)
    )


def validate_receipt(value: Any) -> None:
    """Shape gate for a sealed gate receipt, including the seal itself."""

    if not isinstance(value, Mapping):
        raise DistinctnessError("receipt must be a mapping")
    missing = {
        "schema_version",
        "declared",
        "verified",
        "unverified_model_claims",
        "claims",
        "content_sha256",
    } - set(value)
    if missing:
        raise DistinctnessError(f"receipt missing keys {sorted(missing)}")
    if value["schema_version"] != SCHEMA:
        raise DistinctnessError(f"unexpected schema_version {value['schema_version']!r}")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if canonical_sha256(body) != value["content_sha256"]:
        raise DistinctnessError("receipt seal does not match its content")
    verified = value["verified"]
    counted = int(verified["distinct_proposals"]) + int(verified["rejected"])
    if counted != int(verified["proposals_generated"]):
        raise DistinctnessError(
            f"receipt does not account for every proposal: {counted} != "
            f"{verified['proposals_generated']}"
        )
    for row in value["unverified_model_claims"].values():
        if row.get("status") != "unverified_proposal":
            raise DistinctnessError("model claims must be marked unverified_proposal")


def receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    """Canonical bytes of a sealed receipt, for writing to disk."""

    return canonical_json_bytes(receipt)


__all__ = [
    "DEFAULT_SIGNIFICANT_DIGITS",
    "FINGERPRINT_RULES",
    "GATE_VOCABULARY",
    "PROPOSAL_KINDS",
    "REJECTION_VERDICTS",
    "SCHEMA",
    "UNINFORMATIVE_LEAVES",
    "DistinctnessError",
    "Fingerprint",
    "GateResult",
    "ModelValueLeak",
    "ProbeSuite",
    "Proposal",
    "assert_receipt_is_model_free",
    "constraint_probe",
    "fingerprint_of",
    "generator_probe",
    "model_authored_leaves",
    "problem_probe",
    "program_probe",
    "proposals_from_sources",
    "quantise",
    "receipt_bytes",
    "run_distinctness_gate",
    "validate_receipt",
]
