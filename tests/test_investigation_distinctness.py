"""Gates for the behavioural distinctness gate.

Every positive here has a control that must FAIL, because a gate that only ever admits is
indistinguishable from no gate.  The three the task names are:

* twenty rephrasings of one investigation collapse to one -- built from the four programs a live
  run actually returned, so the headline control is the documented failure and not a mock of it;
* genuinely different investigations survive, in all four proposal kinds;
* a proposal that cannot be run is REJECTED, never counted as a distinct behaviour.

Plus the invariant that outranks them: a model-authored value must never reach a receipt without
passing through exact verification.  The positive is that the real gate's receipt carries none;
the control is a hand-spliced receipt that does, which must raise.
"""

from __future__ import annotations

import json

import pytest

from sigma_theory_compiler import creativity_measure as cm
from sigma_theory_compiler import investigation_distinctness as idg
from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

# The declared response-law domain: 1e-12 to 1e-6, the domain on which the live run's six
# programs were all the identity map.
RESPONSE_POINTS = [10.0 ** (-12 + 0.5 * step) for step in range(13)]


def _program(index: int, body: str, **claims: object) -> idg.Proposal:
    return idg.Proposal(
        f"p{index:02d}", "program", "rule", f"def rule(u):\n    {body}\n", dict(claims)
    )


# The four sources named in creativity_measure's docstring, plus sixteen more spellings of the
# same map.  Nothing here reads the source; the point is that reading it would not help.
IDENTITY_SPELLINGS = (
    "return u",
    "return u / (1 - u/2 + u*u/6)",
    "return u * (1 + u)**(-0.25)",
    "return 2*u / (2 + u)",
    "return u * (1 - u/3 + u*u/12)",
    "return u * 1.0",
    "return (u + u) / 2",
    "return u + 0.0",
    "return sum([u])",
    "return abs(u)",
    "return max(u, 0.0)",
    "return min(u, 1.0)",
    "return u ** 1.0",
    "import math\n    return math.sqrt(u*u)",
    "import math\n    return u * math.cos(u)",
    "import math\n    return u / math.exp(u*u)",
    "import math\n    return math.exp(math.log(u))",
    "return float(u)",
    "return (lambda z: z)(u)",
    "total = 0.0\n    for _ in range(4):\n        total += u / 4\n    return total",
)


@pytest.fixture(scope="module")
def rephrasing_batch() -> idg.GateResult:
    probe = idg.program_probe(RESPONSE_POINTS, label="response_law_domain")
    batch = [_program(i, body) for i, body in enumerate(IDENTITY_SPELLINGS)]
    return idg.run_distinctness_gate(batch, probe)


# ---------------------------------------------------------------------------
# Control 1: twenty rephrasings of one investigation collapse to one
# ---------------------------------------------------------------------------


def test_twenty_rephrasings_of_one_investigation_collapse_to_one(
    rephrasing_batch: idg.GateResult,
) -> None:
    """The documented failure, reproduced and then caught."""

    assert rephrasing_batch.proposals_generated == 20
    assert rephrasing_batch.distinct_proposals == 1
    assert rephrasing_batch.rejected == 19
    verified = rephrasing_batch.receipt["verified"]
    assert verified["rejected_by_verdict"]["duplicate_behaviour"] == 19
    assert verified["rejected_by_verdict"]["unrunnable"] == 0
    # The interesting number: twenty proposals bought one investigation.
    assert float(verified["wasted_variation_ratio"]) == pytest.approx(20.0)
    assert float(verified["effective_distinct_investigations"]) == pytest.approx(1.0)


def test_the_collapse_is_behavioural_not_textual(
    rephrasing_batch: idg.GateResult,
) -> None:
    """Source diversity was twenty; a gate reading source text would have admitted twenty."""

    sources = {proposal.source for proposal in (_program(i, b) for i, b in
                                                enumerate(IDENTITY_SPELLINGS))}
    assert len(sources) == 20
    assert rephrasing_batch.distinct_proposals == 1
    digests = {mark.digest for mark in rephrasing_batch.fingerprints}
    assert len(digests) == 1


def test_every_rejection_names_the_survivor_it_duplicates(
    rephrasing_batch: idg.GateResult,
) -> None:
    rows = rephrasing_batch.receipt["verified"]["rejected_detail"]
    assert len(rows) == 19
    assert {row["duplicate_of"] for row in rows} == {"p00"}


# ---------------------------------------------------------------------------
# Control 2: genuinely different investigations survive -- all four kinds
# ---------------------------------------------------------------------------


def test_genuinely_different_programs_all_survive() -> None:
    probe = idg.program_probe([1.0, 2.0, 3.0, 4.0])
    batch = [
        _program(0, "return u"),
        _program(1, "return u * u"),
        _program(2, "return u + 1"),
        _program(3, "return 1.0 / u"),
        _program(4, "return u * u * u - u"),
    ]
    result = idg.run_distinctness_gate(batch, probe)
    assert result.distinct_proposals == 5
    assert result.rejected == 0
    assert float(result.receipt["verified"]["wasted_variation_ratio"]) == pytest.approx(1.0)


def _generator(index: int, body: str) -> idg.Proposal:
    return idg.Proposal(
        f"g{index}", "generator", "generate", f"def generate(k):\n    {body}\n"
    )


def test_generators_are_the_set_they_emit_not_the_order() -> None:
    """Emission order is not part of a generator's identity; the emitted set is."""

    probe = idg.generator_probe(4, 2, label="rank_two_rows")
    forward = _generator(0, "return [k, k*k]")
    reversed_order = _generator(1, "j = 3 - k\n    return [j, j*j]")
    shifted = _generator(2, "return [k + 10, k*k]")
    result = idg.run_distinctness_gate([forward, reversed_order, shifted], probe)
    assert result.distinct_proposals == 2
    assert [row["verdict"] for row in result.receipt["verified"]["rejected_detail"]] == [
        "duplicate_behaviour"
    ]
    assert result.receipt["verified"]["rejected_detail"][0]["duplicate_of"] == "g0"


def _problem(index: int, body: str) -> idg.Proposal:
    return idg.Proposal(f"q{index}", "problem", "score", f"def score(a, b):\n    {body}\n")


def test_problems_are_the_ordering_they_impose_on_a_ladder() -> None:
    """Two problems are the same only when no rung of the ladder separates them."""

    ladder = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 3.0]]
    probe = idg.problem_probe(ladder, label="reference_ladder")
    absolute = _problem(0, "return abs(a - b)")
    same_but_written_differently = _problem(1, "d = a - b\n    return d if d > 0 else -d")
    permuted = _problem(2, "return abs(b - a) + (1 if a > b else 0)")
    result = idg.run_distinctness_gate([absolute, same_but_written_differently, permuted], probe)
    assert result.distinct_proposals == 2
    assert {p.proposal_id for p in result.admitted} == {"q0", "q2"}


def _constraint(index: int, body: str) -> idg.Proposal:
    return idg.Proposal(f"c{index}", "constraint", "admits", f"def admits(x):\n    {body}\n")


def test_constraints_are_the_line_they_cut_not_the_number_they_return() -> None:
    panel = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
    probe = idg.constraint_probe(panel, label="sign_panel")
    positive_one = _constraint(0, "return 1.0 if x > 0 else 0.0")
    positive_seven = _constraint(1, "return 0.7 if x > 0 else 0.0")
    nonzero = _constraint(2, "return 1.0 if x != 0 else 0.0")
    result = idg.run_distinctness_gate([positive_one, positive_seven, nonzero], probe)
    assert result.distinct_proposals == 2
    assert {p.proposal_id for p in result.admitted} == {"c0", "c2"}


# ---------------------------------------------------------------------------
# Control 3: a proposal that cannot be run is rejected, not counted as distinct
# ---------------------------------------------------------------------------


def test_an_unrunnable_proposal_is_rejected_not_counted_as_distinct() -> None:
    probe = idg.program_probe([1.0, 2.0])
    batch = [
        _program(0, "return u"),
        _program(1, "return ("),
        _program(2, "return 1/0"),
        _program(3, "return 'text'"),
        _program(4, "import os\n    return u"),
        _program(5, "while True:\n        pass"),
        _program(6, "return u + undefined_name"),
    ]
    result = idg.run_distinctness_gate(batch, probe, budget=idg.SandboxBudget(wall_seconds=1.0))
    assert result.distinct_proposals == 1
    assert result.receipt["verified"]["rejected_by_verdict"]["unrunnable"] == 6
    reasons = result.receipt["verified"]["unrunnable_by_sandbox_reason"]
    assert sum(reasons.values()) == 6
    assert set(reasons) <= set(idg.SANDBOX_FAILURE_REASONS)
    assert {p.proposal_id for p in result.admitted} == {"p00"}


def test_failures_do_not_cluster_into_one_distinct_way_of_failing() -> None:
    """Two proposals that both fail to parse are not two ways of investigating anything."""

    probe = idg.program_probe([1.0, 2.0])
    batch = [_program(0, "return ("), _program(1, "return ("), _program(2, "return (")]
    result = idg.run_distinctness_gate(batch, probe)
    assert result.distinct_proposals == 0
    assert result.rejected == 3
    assert result.receipt["verified"]["rejected_by_verdict"]["duplicate_behaviour"] == 0
    digests = {mark.digest for mark in result.fingerprints}
    assert len(digests) == 3, "failure digests must not collide, or failures would cluster"
    assert float(result.receipt["verified"]["wasted_variation_ratio"]) == pytest.approx(0.0)


def test_a_hostile_proposal_is_a_typed_rejection_not_a_distinct_behaviour() -> None:
    probe = idg.program_probe([1.0, 2.0])
    batch = [
        _program(0, "return u"),
        _program(1, "return u.__class__.__mro__[0]"),
        _program(2, "return open('x').read()"),
        _program(3, "return eval('1')"),
    ]
    result = idg.run_distinctness_gate(batch, probe)
    assert result.distinct_proposals == 1
    assert result.receipt["verified"]["rejected_by_verdict"]["unrunnable"] == 3


# ---------------------------------------------------------------------------
# Incumbents: a "new" investigation that behaves like one the system already has
# ---------------------------------------------------------------------------


def test_a_proposal_that_reproduces_an_incumbent_is_rejected() -> None:
    """SUPPORTED_GENERATORS has two entries; a third that behaves like one is not a third."""

    probe = idg.generator_probe(5, 2)
    incumbent = idg.Proposal(
        "riemann_tensor", "generator", "generate", "def generate(k):\n    return [k, 2*k]\n"
    )
    rewrite = idg.Proposal(
        "proposed_a",
        "generator",
        "generate",
        "def generate(k):\n    return [k*1.0, k + k]\n",
        {"description": "a completely new basis generator"},
    )
    genuine = idg.Proposal(
        "proposed_b", "generator", "generate", "def generate(k):\n    return [k, 3*k]\n"
    )
    result = idg.run_distinctness_gate([rewrite, genuine], probe, incumbents=[incumbent])
    assert result.distinct_proposals == 1
    assert {p.proposal_id for p in result.admitted} == {"proposed_b"}
    row = result.receipt["verified"]["rejected_detail"][0]
    assert row["verdict"] == "duplicate_of_incumbent"
    assert row["duplicate_of"] == "riemann_tensor"


def test_an_incumbent_that_cannot_run_stops_the_gate() -> None:
    """Control: a broken baseline is a caller error, never a silently skipped comparison."""

    probe = idg.generator_probe(3, 1)
    broken = idg.Proposal("incumbent_x", "generator", "generate", "def generate(k):\n    return (\n")
    good = idg.Proposal("proposed_a", "generator", "generate", "def generate(k):\n    return k\n")
    with pytest.raises(idg.DistinctnessError, match="broken baseline"):
        idg.run_distinctness_gate([good], probe, incumbents=[broken])


# ---------------------------------------------------------------------------
# The invariant: model output is a proposal, never a result
# ---------------------------------------------------------------------------


POISONED_CLAIMS = {
    "claimed_fingerprint": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "claimed_dimension": "1/2",
    "claimed_verdict": "this investigation is distinct and correct",
    "distinct": True,
    "self_reported_score": 0.997,
    "nested": {"authors_own_novelty": "unprecedented"},
}


@pytest.fixture(scope="module")
def poisoned_batch() -> tuple[list[idg.Proposal], idg.GateResult]:
    probe = idg.program_probe([1.0, 2.0, 3.0])
    batch = [
        _program(0, "return u", **POISONED_CLAIMS),
        _program(1, "return u * 1.0", **POISONED_CLAIMS),
        _program(2, "return u * 5", **POISONED_CLAIMS),
    ]
    return batch, idg.run_distinctness_gate(batch, probe)


def test_no_model_authored_value_reaches_the_receipt(
    poisoned_batch: tuple[list[idg.Proposal], idg.GateResult],
) -> None:
    batch, result = poisoned_batch
    idg.assert_receipt_is_model_free(result.receipt, batch)
    blob = canonical_json_bytes(result.receipt).decode("utf-8")
    for needle in ("1/2", "unprecedented", "0.997", "distinct and correct", "e3b0c442"):
        assert needle not in blob, f"model text {needle!r} survived into the receipt"
    for proposal in batch:
        assert proposal.source not in blob


def test_model_text_appears_only_as_a_digest_marked_unverified(
    poisoned_batch: tuple[list[idg.Proposal], idg.GateResult],
) -> None:
    batch, result = poisoned_batch
    zone = result.receipt["unverified_model_claims"]
    assert set(zone) == {p.proposal_id for p in batch}
    for proposal in batch:
        row = zone[proposal.proposal_id]
        assert row["status"] == "unverified_proposal"
        assert row["source_sha256"] == proposal.source_sha256
        assert row["claims_sha256"] == proposal.claims_sha256
        assert set(row) == {"status", "source_sha256", "claims_sha256"}


def test_a_receipt_that_carries_a_model_value_is_refused(
    poisoned_batch: tuple[list[idg.Proposal], idg.GateResult],
) -> None:
    """THE CONTROL THAT MUST FAIL.

    This is the leaky gate: it does everything the real gate does and then copies one model
    claim into the verified zone, exactly as a careless edit would.  If this does not raise, the
    invariant is unenforced and every test above is decoration.
    """

    batch, result = poisoned_batch
    leaky = json.loads(json.dumps(result.receipt))
    leaky["verified"]["admitted"][0]["dimension"] = POISONED_CLAIMS["claimed_dimension"]
    with pytest.raises(idg.ModelValueLeak, match="model-authored value reached the receipt"):
        idg.assert_receipt_is_model_free(leaky, batch)


def test_a_free_text_field_in_the_verified_zone_is_refused(
    poisoned_batch: tuple[list[idg.Proposal], idg.GateResult],
) -> None:
    """Second control: the channel is closed, not merely the one value that used it."""

    batch, result = poisoned_batch
    leaky = json.loads(json.dumps(result.receipt))
    leaky["verified"]["commentary"] = "the proposer explained that this basis is new"
    with pytest.raises(idg.ModelValueLeak, match="undeclared free text"):
        idg.assert_receipt_is_model_free(leaky, batch)


def test_a_model_claiming_distinctness_is_still_rejected_when_it_duplicates(
    poisoned_batch: tuple[list[idg.Proposal], idg.GateResult],
) -> None:
    """The model does not adjudicate its own distinctness; running it does."""

    _, result = poisoned_batch
    assert result.distinct_proposals == 2
    assert result.receipt["verified"]["rejected_by_verdict"]["duplicate_behaviour"] == 1
    assert result.receipt["claims"]["model_output_is_a_proposal_never_a_result"] is True


def test_a_claim_colliding_with_a_computed_value_refuses_rather_than_seals() -> None:
    """The declared direction of the check: refuse, because a coincidence is indistinguishable.

    A batch of three programs with one behaviour has wasted_variation_ratio "3.000000".  A model
    that claims that exact string cannot be told apart from one whose claim was laundered in, so
    the gate stops instead of sealing.
    """

    probe = idg.program_probe([1.0, 2.0])
    batch = [
        _program(0, "return u", ratio="3.000000"),
        _program(1, "return u * 1.0"),
        _program(2, "return u + 0.0"),
    ]
    with pytest.raises(idg.ModelValueLeak):
        idg.run_distinctness_gate(batch, probe)


def test_uninformative_claims_are_not_treated_as_leaks() -> None:
    """A model asserting `true` or `0` has asserted nothing that could be mistaken for a result."""

    probe = idg.program_probe([1.0, 2.0])
    batch = [
        _program(0, "return u", ok=True, count=0, kind="program", verdict="duplicate_behaviour"),
        _program(1, "return u * 9"),
    ]
    result = idg.run_distinctness_gate(batch, probe)
    assert result.distinct_proposals == 2


# ---------------------------------------------------------------------------
# Resolution is declared, never hidden
# ---------------------------------------------------------------------------


def test_the_documented_six_collapse_at_six_digits_and_separate_at_nine() -> None:
    """Both answers are correct answers to different questions, and the gate says which it used."""

    sources = IDENTITY_SPELLINGS[:5]
    coarse = idg.program_probe(RESPONSE_POINTS, significant_digits=6)
    fine = idg.program_probe(RESPONSE_POINTS, significant_digits=9)
    batch = [_program(i, body) for i, body in enumerate(sources)]
    at_six = idg.run_distinctness_gate(batch, coarse)
    at_nine = idg.run_distinctness_gate(batch, fine)
    assert at_six.distinct_proposals == 1
    assert at_nine.distinct_proposals == 5
    assert at_six.receipt["declared"]["probe"]["significant_digits"] == 6
    assert at_nine.receipt["declared"]["probe"]["significant_digits"] == 9
    assert at_six.receipt["declared"]["probe_sha256"] != at_nine.receipt["declared"]["probe_sha256"]


def test_a_resolution_dependent_split_is_reported_not_silent() -> None:
    """Two behaviours 3e-7 apart: distinct at eight digits, one behaviour at seven.

    The gate admits both and then says out loud that the split hangs on the declared digit,
    rather than letting a reader take a knife-edge separation for a robust one.
    """

    probe = idg.program_probe([1.0, 10.0, 100.0], significant_digits=8)
    batch = [_program(0, "return u"), _program(1, "return u * (1 + 3e-7)")]
    result = idg.run_distinctness_gate(batch, probe)
    assert result.distinct_proposals == 2
    assert result.receipt["verified"]["coarse_collision_pairs"] == 1

    coarser = idg.program_probe([1.0, 10.0, 100.0], significant_digits=7)
    assert idg.run_distinctness_gate(batch, coarser).distinct_proposals == 1


def test_a_robust_split_reports_no_coarse_collisions() -> None:
    probe = idg.program_probe([1.0, 2.0, 3.0])
    batch = [_program(0, "return u"), _program(1, "return u * 100")]
    result = idg.run_distinctness_gate(batch, probe)
    assert result.receipt["verified"]["coarse_collision_pairs"] == 0


def test_a_fingerprint_is_bound_to_the_probe_it_was_taken_on() -> None:
    """Fingerprints from different probes are not comparable, and the digest makes that so."""

    proposal = _program(0, "return u")
    left = idg.fingerprint_of(proposal, idg.program_probe([1.0, 2.0]))
    right = idg.fingerprint_of(proposal, idg.program_probe([1.0, 2.0, 3.0]))
    assert left.coordinates != right.coordinates
    assert left.digest != right.digest


# ---------------------------------------------------------------------------
# It is a superset of creativity_measure, not a competing second opinion
# ---------------------------------------------------------------------------


def test_program_kind_agrees_with_the_creativity_measure_it_generalises() -> None:
    probe = idg.program_probe([1.0, 2.0, 3.0])
    batch = [
        _program(0, "return u"),
        _program(1, "return u * 1.0"),
        _program(2, "return u * 2"),
        _program(3, "return u * 2.0"),
        _program(4, "return u * 3"),
    ]
    result = idg.run_distinctness_gate(batch, probe)
    population = [
        {
            "source": proposal.source,
            "origin": "proposed",
            "outputs": list(mark.raw),
            "novelty": {"novelty_multiplier": "1.0"},
        }
        for proposal, mark in zip(batch, result.fingerprints)
    ]
    measured = cm.measure_creativity(population)
    assert measured["population"]["distinct_behaviours"] == result.distinct_proposals == 3
    assert float(measured["wasted_variation_ratio"]) == pytest.approx(
        float(result.receipt["verified"]["wasted_variation_ratio"])
    )


# ---------------------------------------------------------------------------
# Receipt hygiene
# ---------------------------------------------------------------------------


def test_the_receipt_is_exact_and_seals(
    rephrasing_batch: idg.GateResult,
) -> None:
    idg.validate_receipt(rephrasing_batch.receipt)
    # canonical_sha256 forbids floats, so this encoding is the exactness check.
    assert canonical_json_bytes(rephrasing_batch.receipt)
    body = {k: v for k, v in rephrasing_batch.receipt.items() if k != "content_sha256"}
    assert canonical_sha256(body) == rephrasing_batch.receipt["content_sha256"]


def test_a_tampered_receipt_fails_validation(rephrasing_batch: idg.GateResult) -> None:
    """Control: the seal is checked, not decorative."""

    tampered = json.loads(json.dumps(rephrasing_batch.receipt))
    tampered["verified"]["distinct_proposals"] = 20
    with pytest.raises(idg.DistinctnessError):
        idg.validate_receipt(tampered)


def test_every_proposal_is_accounted_for(rephrasing_batch: idg.GateResult) -> None:
    verified = rephrasing_batch.receipt["verified"]
    assert verified["distinct_proposals"] + verified["rejected"] == verified["proposals_generated"]
    assert sum(verified["rejected_by_verdict"].values()) == verified["rejected"]
    assert set(verified["rejected_by_verdict"]) == set(idg.REJECTION_VERDICTS)


def test_the_gate_is_deterministic() -> None:
    probe = idg.program_probe([1.0, 2.0, 3.0])
    batch = [_program(0, "return u"), _program(1, "return u * 2"), _program(2, "return u + u")]
    first = idg.run_distinctness_gate(batch, probe)
    second = idg.run_distinctness_gate(batch, probe)
    assert first.receipt["content_sha256"] == second.receipt["content_sha256"]


def test_the_receipt_names_the_fingerprint_rule_for_the_kind_it_ran() -> None:
    probe = idg.constraint_probe([[1.0], [-1.0]])
    batch = [_constraint(0, "return 1.0 if x > 0 else 0.0")]
    result = idg.run_distinctness_gate(batch, probe)
    rule = result.receipt["declared"]["fingerprint_rule"]
    assert rule == idg.FINGERPRINT_RULES["constraint"]
    assert "admit" in rule["reduction"]


# ---------------------------------------------------------------------------
# Caller errors are caller errors, not silent guesses
# ---------------------------------------------------------------------------


def test_a_kind_mismatch_is_refused_rather_than_guessed() -> None:
    proposal = idg.Proposal("g0", "generator", "generate", "def generate(k):\n    return k\n")
    with pytest.raises(idg.DistinctnessError, match="declares kind"):
        idg.fingerprint_of(proposal, idg.program_probe([1.0]))


def test_an_unsupported_kind_is_refused() -> None:
    with pytest.raises(idg.DistinctnessError, match="not one this gate implements"):
        idg.Proposal("x0", "vibe", "run", "def run():\n    return 1\n")


def test_a_proposal_id_is_a_slug_not_free_text() -> None:
    with pytest.raises(idg.DistinctnessError, match="not a lowercase slug"):
        idg.Proposal("this is a whole sentence", "program", "rule", "def rule(u):\n    return u\n")


def test_an_empty_probe_cannot_distinguish_anything_and_is_refused() -> None:
    with pytest.raises(idg.DistinctnessError, match="cannot distinguish anything"):
        idg.ProbeSuite("program", (), 1)


def test_a_wide_constraint_has_no_declared_admit_bit() -> None:
    with pytest.raises(idg.DistinctnessError, match="admit bit"):
        idg.ProbeSuite("constraint", ((1.0,),), 2)


def test_resolution_outside_the_declared_envelope_is_refused() -> None:
    with pytest.raises(idg.DistinctnessError, match="significant_digits"):
        idg.ProbeSuite("program", ((1.0,),), 1, 0)
    with pytest.raises(idg.DistinctnessError, match="significant_digits"):
        idg.ProbeSuite("program", ((1.0,),), 1, 16)


def test_an_empty_batch_is_zero_distinct_not_an_error() -> None:
    result = idg.run_distinctness_gate([], idg.program_probe([1.0]))
    assert result.distinct_proposals == 0
    assert result.rejected == 0
    idg.validate_receipt(result.receipt)


def test_quantisation_folds_negative_zero() -> None:
    assert idg.quantise("-0.0", 6) == idg.quantise("0.0", 6)
    assert idg.quantise("1.0000005e-06", 6) == idg.quantise("1e-06", 6)
    assert idg.quantise("1.0000005e-06", 9) != idg.quantise("1e-06", 9)


def test_proposal_ids_are_assigned_by_the_gate_not_by_the_model() -> None:
    """A model returns source code and nothing else; everything that labels it is assigned here."""

    sources = ["def rule(u):\n    return u\n", "def rule(u):\n    return u * 2\n"]
    proposals = idg.proposals_from_sources(sources, "program", "rule")
    assert [p.proposal_id for p in proposals] == ["proposed_000", "proposed_001"]
    result = idg.run_distinctness_gate(proposals, idg.program_probe([1.0, 2.0]))
    assert result.distinct_proposals == 2
    assert set(result.receipt["unverified_model_claims"]) == {"proposed_000", "proposed_001"}


# ---------------------------------------------------------------------------
# Sources a live model actually returned, pinned as a regression
# ---------------------------------------------------------------------------

#: Verbatim output from one live haiku call on funsearch_loop's declared blinded_response_law
#: problem (prompt sha256 ef39b3ed353ecee7...).  Three of these read as different corrections to
#: the seed and are the seed over the domain the problem declares: every correction term is at
#: most 5e-9 relative there, two orders below the declared six-digit resolution.  The fourth
#: genuinely departs from it.  Pinned because this is the failure mode in the wild, not a
#: constructed illustration of it.
LIVE_SEED_ANCHORED_SOURCES = {
    "live_a": "def rule(u):\n    return u / (1 - 5e-6 * u**0.5)\n",
    "live_b": "def rule(u):\n    return u / (1 - 3e-6 * u**0.5 / (1 + u**0.5))\n",
    "live_c": "def rule(u):\n    return u / (1 - 4e-6 * (u**0.5) / (1 + 100 * u))\n",
    "live_d": "def rule(u):\n    return u * (1 + 1e-5 / (u**0.5 + 1e-5))\n",
}


def test_live_model_corrections_to_a_seed_are_caught_as_the_seed() -> None:
    """Four live proposals, four distinct sources, two behaviours -- and the gate says two."""

    probe = idg.program_probe(RESPONSE_POINTS, label="blinded_response_law_domain")
    seed = idg.Proposal("seed_identity", "program", "rule", "def rule(u):\n    return u\n")
    batch = [
        idg.Proposal(name, "program", "rule", source)
        for name, source in LIVE_SEED_ANCHORED_SOURCES.items()
    ]
    assert len({p.source for p in batch}) == 4

    without_seed = idg.run_distinctness_gate(batch, probe)
    assert without_seed.distinct_proposals == 2

    with_seed = idg.run_distinctness_gate(batch, probe, incumbents=[seed])
    assert with_seed.distinct_proposals == 1
    assert {p.proposal_id for p in with_seed.admitted} == {"live_d"}
    verdicts = with_seed.receipt["verified"]["rejected_by_verdict"]
    assert verdicts["duplicate_of_incumbent"] == 3
    assert verdicts["duplicate_behaviour"] == 0
    assert {
        row["duplicate_of"] for row in with_seed.receipt["verified"]["rejected_detail"]
    } == {"seed_identity"}


def test_the_live_verdict_is_a_function_of_the_declared_resolution() -> None:
    """Control: the same four live proposals are 1, 2 or 4 investigations, by declared digit.

    Six digits calls three of them the seed.  Nine separates them from the seed but not from
    each other.  Twelve separates all four.  None of those is the true answer -- there is no
    resolution-free fact about behaviour to be right about -- so the gate reports the digit it
    used and lets the caller own the choice.  A gate that hid the number would be asserting the
    fact that does not exist.
    """

    seed = idg.Proposal("seed_identity", "program", "rule", "def rule(u):\n    return u\n")
    batch = [
        idg.Proposal(name, "program", "rule", source)
        for name, source in LIVE_SEED_ANCHORED_SOURCES.items()
    ]
    ladder = {}
    for digits in (6, 9, 12):
        probe = idg.program_probe(RESPONSE_POINTS, significant_digits=digits)
        result = idg.run_distinctness_gate(batch, probe, incumbents=[seed])
        assert result.receipt["declared"]["probe"]["significant_digits"] == digits
        ladder[digits] = result.distinct_proposals
    assert ladder == {6: 1, 9: 2, 12: 4}
