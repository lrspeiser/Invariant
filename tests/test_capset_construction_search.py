"""Gates for the construction problem with an exact computable objective.

The module claims something narrow and checkable: a program is scored by the *measured* size
of the object it builds, never by a claimed one, and an object that fails the constraint scores
exactly zero.  Every test here is organised around one of the four ways that claim could be
false.

*The verifier could be missing violations.*  So the exhaustive scan is checked against an
independent enumeration of every line, the pair count is compared with ``m (m - 1) / 2``
computed separately, and each of the five typed invalidity reasons is produced on purpose.
Every positive here has a control that must fail: the sealed witnesses verify, and a
one-coordinate perturbation of each must not.

*The score could be a claim rather than a measurement.*  So a program that returns a valid set
plus a duplicate, one that returns a point out of range, one that returns a non-integer, and
one that returns a set containing a line are all executed for real and all must score ``0/1``
-- and the receipt validator must refuse a hand-edited record that gives an invalid
construction a nonzero score.

*The novelty channel could be matching text.*  So the probe that finds the sealed witness by
depth-first search -- containing none of its coordinates anywhere in its bytes -- must be
zeroed exactly like the one that writes the list down, while a 20-point construction that is
affinely but not monomially equivalent to the witness must NOT be zeroed, because the module
claims a subgroup test and nothing more.

*The blindness guard could be word-only.*  So the digit-channel guard is exercised directly: a
prompt carrying the sealed record cardinality as a bare numeral is refused, one carrying it as
a substring of a larger number is not, and the declared problems are checked to carry neither.

Plus the honesty core the rest of the repository shares: exact rational arithmetic with no
float on a certificate path, sealed receipts whose seal moves when a byte does, a control that
differs in exactly one declared field, and a replay that reproduces every sealed score.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler import capset_construction_search as cc
from sigma_theory_compiler.funsearch_loop import (
    LoopConfig,
    MockMutationProposer,
    SandboxBudget,
    SpendGovernor,
)
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_json_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]

FAST_SANDBOX = SandboxBudget(wall_seconds=8.0, memory_bytes=256 * 1024 * 1024)

#: A tiny loop.  Every campaign test runs the real sandbox, so the generation count is what
#: keeps the file inside a CI slice rather than a mock anywhere in the pipeline.
TINY = LoopConfig(
    islands=2,
    island_capacity=5,
    generations=4,
    examples_per_prompt=2,
    proposals_per_call=3,
    reset_period=2,
    temperature=0.12,
    seed=20260819,
)


def _fast(problem: cc.ConstructionProblem) -> cc.ConstructionProblem:
    return replace(problem, sandbox=FAST_SANDBOX)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory: pytest.TempPathFactory) -> dict:
    directory = tmp_path_factory.mktemp("capset")
    return cc.run_campaign(
        config=TINY,
        ledger_path=directory / "ledger.json",
        proposer_kind="mock",
        include_hostile_suite=False,
    )


# ---------------------------------------------------------------------------
# The exact geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", [1, 2, 3, 4, 5])
def test_line_enumeration_matches_the_counting_formula(dimension: int) -> None:
    lines = cc.enumerate_lines(dimension)
    assert len(lines) == cc.lines_in(dimension) == 3**dimension * (3**dimension - 1) // 6
    assert len(set(lines)) == len(lines)
    for triple in lines:
        assert triple == tuple(sorted(triple))
        assert cc.third_point(triple[0], triple[1], dimension) == triple[2]
        assert cc.third_point(triple[0], triple[2], dimension) == triple[1]
        assert cc.third_point(triple[1], triple[2], dimension) == triple[0]


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_third_point_is_the_only_completion(dimension: int) -> None:
    size = cc.points_in(dimension)
    digits = cc.digit_table(dimension)
    for left in range(0, size, 7):
        for right in range(left + 1, size, 11):
            other = cc.third_point(left, right, dimension)
            assert other not in (left, right)
            for index in range(dimension):
                total = digits[left][index] + digits[right][index] + digits[other][index]
                assert total % 3 == 0


@pytest.mark.parametrize("dimension", [1, 2, 3, 4, 5])
def test_the_upper_bound_is_proved_by_an_exhibited_partition(dimension: int) -> None:
    certificate = cc.upper_bound_certificate(dimension)
    assert certificate["verified"]
    assert certificate["every_part_is_a_line"]
    assert certificate["covers_every_point_exactly_once"]
    assert certificate["points_covered"] == 3**dimension
    assert certificate["parts"] == 3 ** (dimension - 1)
    assert certificate["elementary_upper_bound"] == 2 * 3 ** (dimension - 1)
    assert cc.elementary_upper_bound(dimension) == 2 * 3 ** (dimension - 1)


def test_a_partition_that_is_not_a_partition_fails_the_bound_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The false control for the bound: exhibit parts that overlap and the certificate refuses."""

    def broken(dimension: int) -> tuple[tuple[int, int, int], ...]:
        return ((0, 9, 18),) * 9

    monkeypatch.setattr(cc, "parallel_class", broken)
    certificate = cc.upper_bound_certificate(3)
    assert not certificate["verified"]
    assert not certificate["covers_every_point_exactly_once"]
    with pytest.raises(cc.CapsetError):
        cc.elementary_upper_bound(3)


def test_dimension_outside_the_declared_window_is_refused() -> None:
    with pytest.raises(cc.CapsetError):
        cc.points_in(0)
    with pytest.raises(cc.CapsetError):
        cc.points_in(cc.MAX_DIMENSION + 1)


# ---------------------------------------------------------------------------
# The verifier is exhaustive, and its positives all carry a failing control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
def test_every_sealed_witness_verifies_exhaustively(dimension: int) -> None:
    record = cc.SEALED_RECORDS[dimension]
    witness = tuple(record["witness"])
    certificate = cc.verify_cap(witness, dimension, max_points=cc.points_in(dimension))
    assert certificate.valid
    assert certificate.cardinality == record["cardinality"] == len(witness)
    assert certificate.pairs_examined == certificate.pairs_expected
    assert certificate.pairs_expected == len(witness) * (len(witness) - 1) // 2
    assert certificate.violating_pairs == 0


@pytest.mark.parametrize("dimension", [3, 4])
def test_perturbing_a_sealed_witness_must_fail(dimension: int) -> None:
    """The control for the test above: move one coordinate and verification must refuse.

    Every point outside the witness is tried in every slot, so this is not one lucky
    perturbation -- it is the exhaustive statement that the witness is maximal in the strong
    sense that no single substitution keeps it valid at full size.
    """

    witness = list(cc.SEALED_RECORDS[dimension]["witness"])
    size = cc.points_in(dimension)
    survivors = []
    for slot in range(len(witness)):
        for replacement in range(size):
            if replacement in witness:
                continue
            perturbed = list(witness)
            perturbed[slot] = replacement
            certificate = cc.verify_cap(perturbed, dimension, max_points=size)
            if certificate.valid:
                survivors.append((slot, replacement))
    assert survivors == [], f"a perturbed witness still verified: {survivors[:5]}"


@pytest.mark.parametrize("dimension", [3, 4])
def test_adding_any_point_to_a_sealed_witness_must_fail(dimension: int) -> None:
    """The witness is maximal: every extension is rejected, and the reason is typed."""

    witness = list(cc.SEALED_RECORDS[dimension]["witness"])
    size = cc.points_in(dimension)
    for extra in range(size):
        if extra in witness:
            continue
        certificate = cc.verify_cap([*witness, extra], dimension, max_points=size)
        assert not certificate.valid
        assert certificate.reason == "contains_a_forbidden_triple"
        assert certificate.first_forbidden_triple is not None
        assert extra in certificate.first_forbidden_triple


def test_the_verifier_agrees_with_an_independent_line_enumeration() -> None:
    """Exhaustive means exhaustive: the pair scan and a full line scan must never disagree."""

    dimension = 3
    lines = set(cc.enumerate_lines(dimension))
    size = cc.points_in(dimension)
    rng_free_samples = [
        [0, 1, 2],
        [0, 1, 3],
        list(range(9)),
        list(range(0, size, 2)),
        list(cc.WITNESS_DIMENSION_3),
        [4, 13, 22, 0],
    ]
    for points in rng_free_samples:
        certificate = cc.verify_cap(points, dimension, max_points=size)
        chosen = set(points)
        expected = sum(1 for line in lines if set(line) <= chosen)
        if len(chosen) != len(points):
            assert certificate.reason == "duplicate_point"
            continue
        assert certificate.violating_pairs == 3 * expected
        assert certificate.to_dict()["forbidden_triples_found"] == expected
        assert certificate.valid == (expected == 0)
        assert certificate.pairs_examined == len(points) * (len(points) - 1) // 2


def test_every_typed_invalidity_reason_is_reachable() -> None:
    size = cc.points_in(3)
    cases = {
        "contains_a_forbidden_triple": [0, 1, 2],
        "duplicate_point": [0, 1, 1],
        "point_out_of_range": [0, 1, size],
        "too_many_points": list(range(size)) + [0],
    }
    for reason, points in cases.items():
        certificate = cc.verify_cap(points, 3, max_points=size)
        assert not certificate.valid
        assert certificate.reason == reason
    assert cc.read_points(("1.5",)) == ([], "non_integer_point")
    assert set(cases) | {"non_integer_point"} == set(cc.INVALID_REASONS)


def test_a_certificate_cannot_claim_validity_without_examining_every_pair() -> None:
    with pytest.raises(cc.CapsetError):
        cc.CapCertificate(3, True, "", 4, 3, 6, 0, None, (0, 1, 3, 4))
    with pytest.raises(cc.CapsetError):
        cc.CapCertificate(3, False, "not_a_declared_reason", 0, 0, 0, 0, None, ())


# ---------------------------------------------------------------------------
# An invalid construction scores exactly zero
# ---------------------------------------------------------------------------


def test_an_invalid_program_scores_zero_end_to_end() -> None:
    """The headline control: a program returning a forbidden triple must score ``0/1``."""

    problem = _fast(cc.declared_problems()["capset_dimension_4"])
    scored = cc.score_construction(problem, cc.invalid_probe_program(4), origin="probe")
    assert scored.sandbox.ok, scored.sandbox.reason
    assert scored.certificate is not None
    assert not scored.certificate.valid
    assert scored.certificate.reason == "contains_a_forbidden_triple"
    assert scored.quality == Fraction(0)
    assert scored.final == Fraction(0)
    assert scored.novelty_reason == "invalid_construction"


def test_one_extra_point_turns_the_best_possible_score_into_zero() -> None:
    """The minimal perturbation control: the sealed witness scores, the witness plus one does not."""

    problem = _fast(cc.declared_problems()["capset_dimension_4_open_orbit"])
    good = cc.score_construction(problem, cc.literal_witness_program(cc.WITNESS_DIMENSION_4))
    assert good.quality == Fraction(20, 54)
    assert good.final == Fraction(20, 54)
    for extra in (2, 5, 80):
        source = cc.literal_witness_program([*cc.WITNESS_DIMENSION_4, extra])
        bad = cc.score_construction(problem, source)
        assert bad.certificate is not None and not bad.certificate.valid
        assert bad.quality == Fraction(0)
        assert bad.final == Fraction(0)


def test_a_program_cannot_claim_a_size_it_did_not_build() -> None:
    """Padding with repeats does not buy cardinality; it invalidates the whole return."""

    problem = _fast(cc.declared_problems()["capset_dimension_3"])
    padded = cc.literal_witness_program([*cc.WITNESS_DIMENSION_3, *cc.WITNESS_DIMENSION_3])
    scored = cc.score_construction(problem, padded)
    assert scored.certificate is not None
    assert scored.certificate.reason == "duplicate_point"
    assert scored.certificate.cardinality == 18
    assert scored.quality == Fraction(0)
    assert scored.final == Fraction(0)


def test_a_program_returning_a_float_or_an_out_of_range_point_scores_zero() -> None:
    problem = _fast(cc.declared_problems()["capset_dimension_3"])
    fractional = cc.score_construction(problem, "def build():\n    return [0, 1.5]\n")
    assert fractional.certificate is not None
    assert fractional.certificate.reason == "non_integer_point"
    assert fractional.final == Fraction(0)
    outside = cc.score_construction(problem, "def build():\n    return [0, 27]\n")
    assert outside.certificate is not None
    assert outside.certificate.reason == "point_out_of_range"
    assert outside.final == Fraction(0)


def test_a_program_that_does_not_execute_scores_zero_with_a_typed_reason() -> None:
    problem = _fast(cc.declared_problems()["capset_dimension_3"])
    scored = cc.score_construction(problem, "def build():\n    return open('x')\n")
    assert not scored.sandbox.ok
    assert scored.certificate is None
    assert scored.final == Fraction(0)
    assert scored.novelty_reason == "not_executable"


def test_the_seed_is_valid_and_deliberately_weak() -> None:
    for name in ("capset_dimension_3", "capset_dimension_4", "capset_dimension_5"):
        problem = _fast(cc.declared_problems()[name])
        scored = cc.score_construction(problem, problem.seed_program(), origin="seed")
        assert scored.sandbox.ok, scored.sandbox.reason
        assert scored.certificate is not None and scored.certificate.valid
        assert scored.certificate.cardinality == 1
        assert scored.cardinality < int(cc.SEALED_RECORDS[problem.dimension]["cardinality"])


def test_the_seed_carries_no_power_of_three_as_a_literal() -> None:
    """``a // 3 // 3 % 3``, not ``a // 9 % 3``: at n=3 the second form is the sealed record."""

    for name, problem in cc.declared_problems().items():
        numerals = {int(token) for token in re.findall(r"\d+", problem.seed_program())}
        assert numerals <= {0, 1, 3, problem.points}, f"{name} carries {sorted(numerals)}"
        assert cc.numeral_violations(problem.seed_program(), problem.forbidden_numerals()) == []


def test_relaxing_every_digit_threshold_is_the_climb_the_seed_opens() -> None:
    """The seed must sit at the bottom of a monotone ladder, or the objective is not climbable."""

    problem = _fast(cc.declared_problems()["capset_dimension_4_open_orbit"])
    ladder = []
    for relaxed in range(5):
        parts = [
            f"{problem.digit_expression(index)} < {2 if index < relaxed else 1}"
            for index in range(4)
        ]
        body = " and ".join(parts)
        source = f"def build():\n    return [a for a in range(81) if {body}]\n"
        scored = cc.score_construction(problem, source)
        assert scored.certificate is not None and scored.certificate.valid
        ladder.append(scored.certificate.cardinality)
    assert ladder == [1, 2, 4, 8, 16]
    over = problem.seed_program().replace("< 1", "< 3")
    broken = cc.score_construction(problem, over)
    assert broken.certificate is not None and not broken.certificate.valid
    assert broken.final == Fraction(0)


def test_the_declared_mutation_grammar_actually_climbs(tmp_path: Path) -> None:
    """C6 made local: a search whose proposal distribution never moves is a lottery.

    The mock mutator is not a model, and this is not a claim about one.  It is the weaker but
    necessary statement that the declared objective, seed and mutation bank compose into
    something a blind rewriter can climb -- measured, on the real sandbox, against the seed it
    started from.
    """

    problem = _fast(cc.declared_problems()["capset_dimension_4"])
    config = replace(
        cc.CAMPAIGN_CONFIG,
        generations=24,
        islands=2,
        island_capacity=6,
        proposals_per_call=4,
        reset_period=4,
    )
    governor = SpendGovernor(tmp_path / "ledger.json", 200, 200, 1)
    block = cc.run_construction_problem(
        problem, config, MockMutationProposer(config.seed, problem.mutation_bank), governor
    )
    seed_cardinality = 1
    reached = block["headline"]["best_verified_cardinality_from_the_search"]
    assert reached > seed_cardinality, "the declared grammar never improved on its own seed"
    assert reached >= 4, f"the climb stalled at {reached}"
    rising = [row["best_verified_cardinality_in_island"] for row in block["population_history"]]
    assert max(rising) >= reached


def test_the_score_is_the_verified_size_over_the_proved_bound() -> None:
    problem = _fast(cc.declared_problems()["capset_dimension_4_open_orbit"])
    source = (
        "def build():\n"
        "    points = []\n"
        "    for a in range(81):\n"
        "        top = 0\n"
        "        rest = a\n"
        "        for k in range(4):\n"
        "            d = rest % 3\n"
        "            if d > top:\n"
        "                top = d\n"
        "            rest = rest // 3\n"
        "        if top == 1:\n"
        "            points.append(a)\n"
        "    return points\n"
    )
    scored = cc.score_construction(problem, source)
    assert scored.certificate is not None and scored.certificate.valid
    assert scored.certificate.cardinality == 15
    assert scored.quality == Fraction(15, 54) == Fraction(5, 18)
    assert scored.final == Fraction(5, 18)


# ---------------------------------------------------------------------------
# The novelty channel measures behaviour, and only claims what it can prove
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dimension", [3, 4])
def test_the_orbit_is_the_declared_group_applied_exhaustively(dimension: int) -> None:
    witness = tuple(cc.SEALED_RECORDS[dimension]["witness"])
    orbit = cc.monomial_orbit(witness, dimension)
    assert cc.points_mask(witness) in orbit
    assert 0 < len(orbit) <= cc.monomial_group_order(dimension)
    assert cc.monomial_group_order(dimension) % len(orbit) == 0
    size = cc.points_in(dimension)
    for mask in orbit:
        points = [point for point in range(size) if (mask >> point) & 1]
        assert len(points) == len(witness)
        assert cc.verify_cap(points, dimension, max_points=size).valid


@pytest.mark.parametrize(
    "label", ["probe_known_literal", "probe_known_translated", "probe_known_scaled"]
)
def test_a_disguised_reproduction_is_zeroed(label: str) -> None:
    problem = _fast(cc.declared_problems()["capset_dimension_4"])
    source = dict(cc.probe_programs(4))[label]
    scored = cc.score_construction(problem, source, origin=label)
    assert scored.certificate is not None and scored.certificate.valid
    assert scored.certificate.cardinality == 20
    assert scored.quality == Fraction(20, 54)
    assert scored.novelty_multiplier == Fraction(0)
    assert scored.final == Fraction(0)
    assert scored.novelty_reason == "monomial_orbit_of_the_sealed_witness"


def test_a_reproduction_that_never_writes_the_answer_down_is_zeroed_too() -> None:
    """The behavioural test: the depth-first probe contains no witness coordinate at all."""

    problem = _fast(cc.declared_problems()["capset_dimension_4"])
    source = dict(cc.probe_programs(4))["probe_known_searched"]
    numerals = {int(token) for token in re.findall(r"\d+", source)}
    absent = [point for point in cc.WITNESS_DIMENSION_4 if point not in numerals]
    assert len(absent) >= 16, f"the searching probe carries the answer: only {absent} are absent"
    scored = cc.score_construction(problem, source, origin="probe_known_searched")
    assert scored.certificate is not None and scored.certificate.valid
    assert tuple(scored.certificate.points) == cc.WITNESS_DIMENSION_4
    assert scored.novelty_multiplier == Fraction(0)
    assert scored.final == Fraction(0)


def test_an_affine_but_not_monomial_equivalent_is_not_zeroed() -> None:
    """The claim ``monomial_orbit_absence_establishes_novelty: False`` made concrete.

    A shear is linear and invertible over ``F_3`` but is not a permutation-times-scaling, so
    its image of the sealed witness is a genuine rediscovery that this channel cannot see.  The
    channel must return a nonzero multiplier here, and the module must not call that novelty.
    """

    dimension = 4
    digits = cc.digit_table(dimension)

    def sheared(code: int) -> int:
        row = list(digits[code])
        row[0] = (row[0] + row[3]) % 3
        return sum(value * 3**index for index, value in enumerate(row))

    image = sorted(sheared(point) for point in cc.WITNESS_DIMENSION_4)
    certificate = cc.verify_cap(image, dimension, max_points=81)
    assert certificate.valid and certificate.cardinality == 20
    assert cc.points_mask(image) not in cc.monomial_orbit(cc.WITNESS_DIMENSION_4, dimension)
    verdict = cc.orbit_novelty(certificate, cc.WITNESS_DIMENSION_4, cc.OrbitPolicy())
    assert verdict["multiplier"] == Fraction(1)
    assert verdict["reason"] == "distance_from_the_sealed_witness_orbit"
    assert cc.CLAIMS["monomial_orbit_absence_establishes_novelty"] is False


def test_the_cardinality_gap_shortcut_agrees_with_the_full_scan() -> None:
    """The shortcut is exact, so it must never disagree with enumerating the orbit."""

    policy = cc.OrbitPolicy(saturation=Fraction(1, 1))
    witness = cc.WITNESS_DIMENSION_3
    for points in ([0], [0, 1], list(witness[:5]), list(witness)):
        certificate = cc.verify_cap(points, 3, max_points=27)
        scanned = cc.orbit_novelty(certificate, witness, policy)
        assert not scanned["detail"].get("orbit_scan_skipped", False)
        mask = cc.points_mask(points)
        best = min((mask ^ image).bit_count() for image in cc.monomial_orbit(witness, 3))
        expected = Fraction(best, len(points) + len(witness))
        assert scanned["multiplier"] == policy.multiplier_from_distance(expected)


def test_an_invalid_construction_is_never_credited_by_the_orbit_channel() -> None:
    certificate = cc.verify_cap([0, 1, 2], 3, max_points=27)
    verdict = cc.orbit_novelty(certificate, cc.WITNESS_DIMENSION_3, cc.OrbitPolicy())
    assert verdict["multiplier"] == Fraction(0)
    assert verdict["reason"] == "invalid_construction"


def test_an_arm_with_no_sealed_witness_says_so_rather_than_claiming_novelty() -> None:
    certificate = cc.verify_cap([1, 3, 9, 27, 81], 5, max_points=243)
    assert certificate.valid
    verdict = cc.orbit_novelty(certificate, (), cc.OrbitPolicy())
    assert verdict["multiplier"] == Fraction(1)
    assert verdict["reason"] == "no_sealed_witness_declared"
    assert cc.SEALED_RECORDS[5]["witness"] == ()


# ---------------------------------------------------------------------------
# Blindness on both channels
# ---------------------------------------------------------------------------


def test_the_numeral_guard_matches_whole_runs_only() -> None:
    assert cc.numeral_violations("build a list of 20 things", [20]) == ["20"]
    assert cc.numeral_violations("range(120)", [20]) == []
    assert cc.numeral_violations("x = 1200 + 20", [20, 54]) == ["20"]
    assert cc.numeral_violations("a[54]", [20, 54]) == ["54"]
    assert cc.numeral_violations("nothing here", [20]) == []


@pytest.mark.parametrize(
    "name",
    [
        "capset_dimension_3",
        "capset_dimension_4",
        "capset_dimension_5",
        "capset_dimension_4_open_orbit",
    ],
)
def test_the_declared_problem_leaks_neither_a_word_nor_a_numeral(name: str) -> None:
    problem = cc.declared_problems()[name]
    prompt = cc.build_construction_prompt(problem, [])
    assert cc.numeral_violations(prompt, problem.forbidden_numerals()) == []
    record = int(cc.SEALED_RECORDS[problem.dimension]["cardinality"])
    assert record in problem.forbidden_numerals()
    assert cc.elementary_upper_bound(problem.dimension) in problem.forbidden_numerals()
    assert str(problem.dimension) in prompt


def test_a_prompt_carrying_the_sealed_numeral_is_refused_not_sanitised() -> None:
    problem = _fast(cc.declared_problems()["capset_dimension_4"])
    leaking = cc.score_construction(
        problem, "def build():\n    return [a for a in range(20) if a % 5 == 0]\n"
    )
    with pytest.raises(cc.CapsetError, match="sealed numeral"):
        cc.build_construction_prompt(problem, [leaking])


def test_a_prompt_carrying_a_forbidden_word_is_refused() -> None:
    problem = _fast(cc.declared_problems()["capset_dimension_4"])
    named = cc.score_construction(problem, "def build():\n    cap = [0, 1]\n    return cap\n")
    with pytest.raises(cc.CapsetError, match="forbidden vocabulary"):
        cc.build_construction_prompt(problem, [named])


def test_a_refused_prompt_is_logged_and_the_loop_continues(tmp_path: Path) -> None:
    """A leak stops that generation; it must not stop the run or vanish from the receipt."""

    problem = _fast(cc.declared_problems()["capset_dimension_4"])
    governor = SpendGovernor(tmp_path / "ledger.json", 40, 40, 1)

    class Leaker:
        proposer_id = "test_leaker"

        def __init__(self) -> None:
            self._last: tuple[str, ...] = ()

        def propose(self, prompt, examples, count):
            self._last = ("def build():\n    return [a for a in range(20) if a % 7 == 0]\n",)
            return cc.ProposalCall(self.proposer_id, "", len(prompt), 1, True, "", "")

        def programs(self) -> tuple[str, ...]:
            return self._last

    block = cc.run_construction_problem(
        problem, replace(TINY, generations=4, islands=1), Leaker(), governor
    )
    assert block["blindness_refusals"], "the leaking example never reached a prompt"
    assert block["headline"]["prompts_refused_by_the_blindness_guard"] >= 1
    reasons = {call["reason"] for call in block["proposal_calls"] if not call["ok"]}
    assert reasons == {"prompt_refused_by_the_blindness_guard"}
    assert block["generations_run"] >= 1


# ---------------------------------------------------------------------------
# Exactness: no float reaches a sealed number
# ---------------------------------------------------------------------------


def test_the_receipt_is_canonicalisable_which_rejects_floats(campaign: dict) -> None:
    """``canonical_json_bytes`` refuses floats, so sealing the receipt at all proves the rule."""

    body = {key: item for key, item in campaign.items() if key != "content_sha256"}
    assert canonical_sha256(body) == campaign["content_sha256"]
    with pytest.raises(SchemaViolation):
        canonical_json_bytes({"quality": 0.5})
    with pytest.raises(SchemaViolation):
        canonical_json_bytes({"nested": [{"final_score": 0.37}]})


def test_every_sealed_score_is_an_exact_rational(campaign: dict) -> None:
    for block in campaign["problems"]:
        for item in block["sealed_programs"]:
            quality = Fraction(item["quality"])
            multiplier = Fraction(item["novelty"]["multiplier"])
            final = Fraction(item["final_score"])
            assert quality * multiplier == final
            assert 0 <= quality <= 1
            assert 0 <= multiplier <= 1


def test_decimal_rendering_is_integer_arithmetic() -> None:
    assert cc._decimal(Fraction(1, 3), 6) == "0.333333"
    assert cc._decimal(Fraction(2, 3), 6) == "0.666667"
    assert cc._decimal(Fraction(20, 54), 9) == "0.370370370"
    assert cc._decimal(Fraction(0), 3) == "0.000"
    assert cc._decimal(Fraction(1), 3) == "1.000"
    with pytest.raises(cc.CapsetError):
        cc._decimal(Fraction(-1, 2))


# ---------------------------------------------------------------------------
# The campaign, its control, and its seals
# ---------------------------------------------------------------------------


def test_the_campaign_validates_and_declares_every_arm(campaign: dict) -> None:
    cc.validate_receipt(campaign)
    labels = [block["run_label"] for block in campaign["problems"]]
    assert set(labels) == set(cc.RUN_LABEL_PROBLEM)
    assert campaign["schema_version"] == cc.RECEIPT_SCHEMA
    assert campaign["claims"] == cc.CLAIMS
    assert campaign["proposer"]["used"] == "deterministic_mock_mutator"


def test_the_record_gate_is_false_and_says_why(campaign: dict) -> None:
    assert campaign["headline"]["any_arm_beats_its_sealed_record"] is False
    assert campaign["headline"]["any_search_beats_its_sealed_record"] is False
    for row in campaign["headline"]["arms"]:
        record = int(cc.SEALED_RECORDS[row["dimension"]]["cardinality"])
        assert row["sealed_record_cardinality"] == record
        assert row["best_verified_cardinality"] <= record
        assert row["beats_sealed_record"] is False


def test_the_search_number_excludes_the_hand_written_probes(campaign: dict) -> None:
    for block in campaign["problems"]:
        searched = [
            item
            for item in block["sealed_programs"]
            if item["origin"] in ("seed", "proposed")
            and item["certificate"]
            and item["certificate"]["valid"]
        ]
        expected = max((item["certificate"]["cardinality"] for item in searched), default=0)
        assert block["headline"]["best_verified_cardinality_from_the_search"] == expected


def test_every_sealed_certificate_is_exhaustive(campaign: dict) -> None:
    seen = 0
    for block in campaign["problems"]:
        for item in block["sealed_programs"]:
            certificate = item["certificate"]
            if certificate is None or not certificate["valid"]:
                continue
            seen += 1
            count = certificate["cardinality"]
            assert certificate["pairs_examined"] == count * (count - 1) // 2
            assert certificate["exhaustive"]
            assert canonical_sha256(certificate["points"]) == certificate["points_sha256"]
            replay = cc.verify_cap(
                certificate["points"],
                certificate["dimension"],
                max_points=block["problem"]["max_points"],
            )
            assert replay.to_dict() == certificate
    assert seen > 0


def test_every_invalid_sealed_program_scored_zero(campaign: dict) -> None:
    seen = 0
    for block in campaign["problems"]:
        for item in block["sealed_programs"]:
            certificate = item["certificate"]
            if certificate is None or certificate["valid"]:
                continue
            seen += 1
            assert certificate["reason"] in cc.INVALID_REASONS
            assert Fraction(item["quality"]) == 0
            assert Fraction(item["final_score"]) == 0
    assert seen > 0, "no invalid construction appeared, so the zero rule was never exercised"


def test_the_control_isolates_the_multiplier(campaign: dict) -> None:
    by_label = {block["run_label"]: block for block in campaign["problems"]}
    live = by_label["capset_dimension_4"]
    control = by_label["capset_dimension_4_open_orbit"]
    assert control["generations_run"] == 0
    assert control["problem"]["sealed_witness_declared"] is False
    assert live["problem"]["sealed_witness_declared"] is True
    assert live["problem"]["elementary_upper_bound"] == control["problem"]["elementary_upper_bound"]

    live_rows = {item["program_sha256"]: item for item in live["sealed_programs"]}
    control_rows = {item["program_sha256"]: item for item in control["sealed_programs"]}
    shared = set(live_rows) & set(control_rows)
    assert shared
    moved = 0
    for digest in shared:
        assert live_rows[digest]["quality"] == control_rows[digest]["quality"]
        assert live_rows[digest]["certificate"] == control_rows[digest]["certificate"]
        if live_rows[digest]["novelty"]["multiplier"] != control_rows[digest]["novelty"][
            "multiplier"
        ]:
            moved += 1
    assert moved >= 1
    assert live["headline"]["constructions_in_the_sealed_orbit"] >= 1
    assert control["headline"]["constructions_in_the_sealed_orbit"] == 0


def test_the_receipt_replays_exactly(campaign: dict) -> None:
    report = cc.replay_from_receipt(campaign)
    assert report["programs_checked"] > 0
    assert report["identical"], report["mismatches"][:3]


@pytest.mark.parametrize(
    "mutation",
    [
        "raise_a_zeroed_score",
        "raise_an_invalid_cardinality",
        "break_the_points_seal",
        "flip_the_record_gate",
        "widen_the_upper_bound",
        "empty_the_live_orbit",
        "change_a_claim",
    ],
)
def test_a_tampered_receipt_is_refused(campaign: dict, mutation: str) -> None:
    """Every gate that can be made green by editing a pin is a gate that is not a gate."""

    value = copy.deepcopy(campaign)
    by_label = {block["run_label"]: block for block in value["problems"]}
    live = by_label["capset_dimension_4"]
    if mutation == "raise_a_zeroed_score":
        target = next(
            item
            for item in live["sealed_programs"]
            if item["novelty"]["reason"] == "monomial_orbit_of_the_sealed_witness"
        )
        target["novelty"]["multiplier"] = "1/1"
    elif mutation == "raise_an_invalid_cardinality":
        target = next(
            item
            for item in live["sealed_programs"]
            if item["certificate"] and not item["certificate"]["valid"]
        )
        target["quality"] = "1/1"
        target["final_score"] = "1/1"
    elif mutation == "break_the_points_seal":
        target = next(
            item
            for item in live["sealed_programs"]
            if item["certificate"] and item["certificate"]["valid"]
            and item["certificate"]["cardinality"] > 2
        )
        target["certificate"]["points"][0] = target["certificate"]["points"][-1]
    elif mutation == "flip_the_record_gate":
        live["headline"]["beats_sealed_record"] = True
    elif mutation == "widen_the_upper_bound":
        live["problem"]["elementary_upper_bound"] += 1
    elif mutation == "empty_the_live_orbit":
        live["problem"]["sealed_witness_declared"] = False
    else:
        value["claims"] = {**value["claims"], "verification_is_exhaustive_not_sampled": False}
    # Re-seal so the tamper is not caught by the outer hash alone: the point is that the
    # *arithmetic* refuses it even when the seals have been recomputed to agree.
    value.pop("content_sha256")
    value.pop("result_core_sha256")
    measurement = value.pop("measurement")
    value["result_core_sha256"] = canonical_sha256(value)
    value["measurement"] = measurement
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    with pytest.raises(cc.CapsetError):
        cc.validate_receipt(value)


def test_editing_one_byte_moves_the_seal(campaign: dict) -> None:
    value = copy.deepcopy(campaign)
    value["problems"][0]["sealed_programs"][0]["origin"] = "tampered"
    with pytest.raises(cc.CapsetError, match="seal changed"):
        cc.validate_receipt(value)


def test_the_sealed_record_is_bound_to_the_table(campaign: dict) -> None:
    for block in campaign["problems"]:
        record = block["sealed_record_revealed_after_scoring"]
        assert canonical_sha256(record) == block["sealed_record_sha256"]
        assert record == cc.sealed_record(block["problem"]["dimension"])
        if record["witness_declared"]:
            certificate = cc.verify_cap(
                record["witness"], record["dimension"], max_points=3 ** record["dimension"]
            )
            assert certificate.valid
            assert certificate.cardinality == record["cardinality"]


def test_the_budget_is_charged_and_ledgered(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    result = cc.run_campaign(
        config=replace(TINY, generations=2, islands=1),
        ledger_path=ledger,
        max_calls=4,
        max_dollars_hundredths=4,
        proposer_kind="mock",
        include_hostile_suite=False,
    )
    cc.validate_receipt(result)
    assert result["budget"]["calls"] <= 4
    assert result["budget"]["halt_reason"] in ("", "call_cap_reached", "dollar_cap_reached")
    assert json.loads(ledger.read_text(encoding="utf-8"))["spent_dollars_hundredths"] == (
        result["budget"]["closing_ledger_hundredths"]
    )


def test_an_unreachable_live_proposer_refuses_rather_than_degrading(tmp_path: Path) -> None:
    with pytest.raises(cc.ProposerUnavailable):
        cc.run_campaign(
            config=replace(TINY, generations=1, islands=1),
            ledger_path=tmp_path / "ledger.json",
            proposer_kind="claude",
            claude_executable=str(tmp_path / "definitely-not-a-binary"),
            include_hostile_suite=False,
        )


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def test_the_cli_writes_validates_and_replays_one_receipt(tmp_path: Path) -> None:
    output = tmp_path / "construction-v1.json"
    common = [
        sys.executable,
        "-m",
        "sigma_theory_compiler.capset_construction_search",
        "--output",
        str(output),
        "--ledger",
        str(tmp_path / "ledger.json"),
    ]
    build = subprocess.run(
        [*common, "--generations", "2", "--islands", "1", "--no-hostile-suite"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert build.returncode == 0, build.stderr[-2000:]
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == cc.RECEIPT_SCHEMA
    validated = subprocess.run(
        [*common, "--validate-checked"], capture_output=True, text=True, check=False, cwd=ROOT
    )
    assert validated.returncode == 0, validated.stderr[-2000:]
    assert json.loads(validated.stdout)["validated"] is True
    replayed = subprocess.run(
        [*common, "--replay-checked"], capture_output=True, text=True, check=False, cwd=ROOT
    )
    assert replayed.returncode == 0, replayed.stderr[-2000:]
    assert json.loads(replayed.stdout)["identical"] is True


def test_the_module_is_registered_for_lint_and_for_ci() -> None:
    workflow = (ROOT / ".github" / "workflows" / "sigma-theory-compiler.yml").read_text(
        encoding="utf-8"
    )
    assert "src/sigma_theory_compiler/capset_construction_search.py" in workflow
    assert "tests/test_capset_construction_search.py" in workflow
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "src/sigma_theory_compiler/capset_construction_search.py text eol=lf" in attributes
    assert "tests/test_capset_construction_search.py text eol=lf" in attributes
