"""Gates for the FunSearch-style loop with anti-recall novelty pressure.

The module is built against one observed failure -- a model asked about galaxy rotation
"rediscovered" MOND by rewriting it -- so the tests are organised around the three things
that have to be true for the mechanism to mean anything.

*The multiplier measures behaviour, not text.*  The MOND probes include two programs that
never write the formula down: one iterates a fixed point to convergence, one bisects a root.
If the grammar match were textual they would sail through.  They must be zeroed, and the
measured distance must sit at float roundoff rather than merely "small".

*The multiplier, not the quality function, is what changes.*  The control differs from the
live problem in exactly one declared field.  Every quality score must be bit-identical
across the pair and the multipliers must move, or the demonstration proves nothing.

*The sandbox classifies, it never crashes.*  Programs are model output, so the hostile suite
is run for real: an infinite loop, two memory bombs, an undeclared import, a file write, the
``__subclasses__`` escape, ``eval``, a syntax error, a division by zero, a non-finite return
and a missing entry point.  Each must come back with a declared typed reason.

Plus the honesty core the rest of the repository shares: sealed receipts, a claims block that
keeps ``corpus_absence_establishes_novelty`` false, budget caps that halt the loop cleanly,
a vocabulary guard that refuses a leaking prompt, and a replay that reproduces every sealed
score exactly from the sealed programs.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sigma_theory_compiler import funsearch_loop as fl

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "runs" / "math" / "funsearch" / "loop-v1.json"
CORPUS_DB = ROOT / "runs" / "math" / "prior-art" / "cf-corpus-v1.sqlite"
CORPUS_MANIFEST = ROOT / "runs" / "math" / "prior-art" / "cf-corpus-v1-manifest.json"

FAST_SANDBOX = fl.SandboxBudget(wall_seconds=4.0, memory_bytes=256 * 1024 * 1024)


@pytest.fixture(scope="module")
def receipt() -> dict:
    if not RECEIPT.exists():
        pytest.skip("receipt not present")
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus():
    if not CORPUS_DB.exists():
        pytest.skip("prior-art corpus not present")
    from sigma_theory_compiler.cf_prior_art_corpus import load_corpus

    return load_corpus(CORPUS_DB, CORPUS_MANIFEST)


# ---------------------------------------------------------------------------
# The sandbox: hostile programs are classified, never fatal
# ---------------------------------------------------------------------------


def test_hostile_suite_contains_every_program() -> None:
    results = fl.run_hostile_suite(FAST_SANDBOX)
    assert len(results) == len(fl.HOSTILE_PROGRAMS)
    escaped = [item for item in results if not item["contained"]]
    assert escaped == [], f"a hostile program escaped the sandbox: {escaped}"
    for item in results:
        assert item["reason"] in fl.SANDBOX_FAILURE_REASONS


def test_hostile_reasons_are_the_expected_ones() -> None:
    reasons = {item["program"]: item["reason"] for item in fl.run_hostile_suite(FAST_SANDBOX)}
    assert reasons["infinite_loop"] == "timeout"
    # A hard limit (RLIMIT_AS, or a Windows job object) fails the allocation itself and
    # surfaces as MemoryError; only the polling fallback can report memory_cap_exceeded.
    assert reasons["memory_bomb_incremental"] in {"memory_cap_exceeded", "runtime_error"}
    assert reasons["memory_bomb_single_allocation"] in {"memory_cap_exceeded", "runtime_error"}
    assert reasons["import_socket"] == "static_screen_denied"
    assert reasons["file_write"] == "static_screen_denied"
    assert reasons["subclass_escape"] == "static_screen_denied"
    assert reasons["eval_escape"] == "static_screen_denied"
    assert reasons["syntax_error"] == "syntax_error"
    assert reasons["divide_by_zero"] == "runtime_error"
    assert reasons["non_finite"] == "non_finite_output"
    assert reasons["wrong_entry"] == "entry_missing"


def test_infinite_loop_is_killed_near_the_declared_wall() -> None:
    budget = fl.SandboxBudget(wall_seconds=1.0, memory_bytes=256 * 1024 * 1024)
    source = fl._program("def rule(u):", "    while True:", "        pass")
    outcome = fl.run_in_sandbox(source, "rule", ((1.0,),), budget)
    assert outcome.reason == "timeout"
    assert outcome.elapsed_seconds < 6.0


def test_static_screen_denies_the_documented_escapes() -> None:
    assert fl.static_screen("import os\ndef rule(u):\n    return u\n", ("math",)) == [
        "import_denied:os"
    ]
    assert "denied_name:getattr" in fl.static_screen(
        "def rule(u):\n    return getattr(u, 'real')\n", ("math",)
    )
    assert "dunder_string_literal" in fl.static_screen(
        "def rule(u):\n    return len('__class__')\n", ("math",)
    )
    assert fl.static_screen("def rule(u):\n    return u + 1\n", ("math",)) == []


def test_allowed_import_still_works() -> None:
    source = fl._program("import math", "def rule(u):", "    return math.sqrt(u)")
    outcome = fl.run_in_sandbox(source, "rule", ((4.0,),), FAST_SANDBOX)
    assert outcome.ok
    assert float(outcome.outputs[0]) == pytest.approx(2.0)


def test_runtime_import_of_an_undeclared_module_is_denied() -> None:
    """The guarded ``__import__`` is a second layer under the static screen, so test it alone."""

    source = fl._program(
        "def rule(u):",
        "    mod = __import__('socket')",
        "    return u",
    )
    assert "denied_name:__import__" in fl.static_screen(source, ("math",))
    inner = fl._program("import math", "def rule(u):", "    return math.pi")
    outcome = fl.run_in_sandbox(inner, "rule", ((1.0,),), fl.SandboxBudget(import_allowlist=()))
    assert outcome.reason == "static_screen_denied"


def test_sandbox_never_raises_on_arbitrary_bytes() -> None:
    for source in ("", "\x00\x01", "def rule(u): return", "rule = 5"):
        outcome = fl.run_in_sandbox(source, "rule", ((1.0,),), FAST_SANDBOX)
        assert not outcome.ok
        assert outcome.reason in fl.SANDBOX_FAILURE_REASONS


# ---------------------------------------------------------------------------
# The vocabulary guard
# ---------------------------------------------------------------------------


def test_vocabulary_guard_tokenises_through_punctuation() -> None:
    forbidden = ("gravity", "mond")
    assert fl.vocabulary_violations("a MOND-like law", forbidden) == ["mond"]
    assert fl.vocabulary_violations("dark_gravity_term", forbidden) == ["gravity"]
    assert fl.vocabulary_violations("a harmless prompt", forbidden) == []


def test_prompt_is_refused_when_a_program_name_leaks_the_domain() -> None:
    """The realistic leak: a proposed program names a variable after the subject."""

    problem = fl.declared_problems()["blinded_response_law"]
    leaky = fl.ScoredProgram(
        "0" * 64,
        "def rule(u):\n    gravity = 2.0\n    return u * gravity\n",
        "proposed",
        0,
        0,
        fl.SandboxOutcome(True),
        (),
        0.5,
        {},
        fl.NoveltyVerdict(1.0, "x", {}),
        0.5,
    )
    with pytest.raises(fl.FunSearchError, match="gravity"):
        fl.build_prompt(problem, (leaky,))


def test_clean_prompt_carries_no_domain_vocabulary() -> None:
    problem = fl.declared_problems()["blinded_response_law"]
    seeded = fl.score_program(problem, problem.seed_program, origin="seed")
    prompt = fl.build_prompt(problem, (seeded,))
    assert fl.vocabulary_violations(prompt, problem.forbidden_vocabulary) == []
    assert "1.2e-10" not in prompt
    for token in ("mond", "gravity", "galaxy", "1.2e-10"):
        assert token not in prompt.lower()


def test_no_declared_problem_leaks_its_own_answer_into_the_prompt_path() -> None:
    for problem in fl.declared_problems().values():
        assert fl.vocabulary_violations(problem.seed_program, problem.forbidden_vocabulary) == []
        assert fl.vocabulary_violations(problem.signature_text, problem.forbidden_vocabulary) == []


# ---------------------------------------------------------------------------
# The anti-recall multiplier -- the point of the module
# ---------------------------------------------------------------------------


def test_every_known_variant_including_disguises_is_zeroed() -> None:
    problem = fl.declared_problems()["blinded_response_law"]
    for label, source in fl.RESPONSE_PROBE_PROGRAMS:
        if not label.startswith("probe_known"):
            continue
        scored = fl.score_program(problem, source, origin=label)
        assert scored.quality == pytest.approx(1.0, abs=1e-9), label
        assert scored.novelty.multiplier == 0.0, label
        assert scored.final_score == 0.0, label
        assert scored.novelty.reason.startswith("known_solution_grammar_match:"), label


def test_the_two_behavioural_disguises_never_write_the_formula() -> None:
    """A textual check would pass these; the behavioural one must not."""

    sources = dict(fl.RESPONSE_PROBE_PROGRAMS)
    for label in ("probe_known_disguise_fixed_point", "probe_known_disguise_bisection"):
        body = sources[label]
        assert "0.5 * u * (1.0 + math.sqrt" not in body
        assert "sqrt(1.0 + 4.0" not in body
    problem = fl.declared_problems()["blinded_response_law"]
    for label in ("probe_known_disguise_fixed_point", "probe_known_disguise_bisection"):
        scored = fl.score_program(problem, sources[label], origin=label)
        distance = float(scored.novelty.detail["nearest_family"]["distance"])
        assert distance < 1e-12, f"{label} distance {distance} is not roundoff"
        assert scored.novelty.multiplier == 0.0


def test_a_genuinely_different_law_keeps_its_multiplier() -> None:
    problem = fl.declared_problems()["blinded_response_law"]
    sources = dict(fl.RESPONSE_PROBE_PROGRAMS)
    scored = fl.score_program(problem, sources["probe_alternative_linear"], origin="probe")
    assert scored.novelty.multiplier == 1.0
    assert scored.final_score == pytest.approx(scored.quality)


def test_multiplier_is_monotone_in_the_measured_distance() -> None:
    policy = fl.NoveltyPolicy()
    assert policy.multiplier_from_distance(0.0) == 0.0
    assert policy.multiplier_from_distance(policy.zero_threshold) == 0.0
    assert 0.0 < policy.multiplier_from_distance(0.05) < 1.0
    assert policy.multiplier_from_distance(0.25) == 1.0
    assert policy.multiplier_from_distance(10.0) == 1.0
    rising = [policy.multiplier_from_distance(item / 100.0) for item in range(30)]
    assert rising == sorted(rising)


def test_novelty_is_a_selection_pressure_not_a_postcheck() -> None:
    """A zeroed program must be unable to win selection, which is what makes it a pressure."""

    problem = fl.declared_problems()["blinded_response_law"]
    sources = dict(fl.RESPONSE_PROBE_PROGRAMS)
    perfect = fl.score_program(problem, sources["probe_known_canonical"], origin="probe")
    worse = fl.score_program(problem, sources["probe_alternative_linear"], origin="probe")
    assert perfect.quality > worse.quality
    assert perfect.final_score < worse.final_score


def test_emptying_the_grammar_is_what_moves_the_score() -> None:
    live = fl.declared_problems()["blinded_response_law"]
    control = fl.declared_problems()["blinded_response_law_control"]
    assert live.evaluator_id == control.evaluator_id
    assert live.probe_points == control.probe_points
    assert live.sandbox == control.sandbox
    assert control.known_solution_grammar == ()
    for label, source in fl.RESPONSE_PROBE_PROGRAMS:
        left = fl.score_program(live, source, origin=label)
        right = fl.score_program(control, source, origin=label)
        assert left.quality == right.quality, label
        assert right.novelty.multiplier == 1.0, label
        if label.startswith("probe_known"):
            assert left.final_score == 0.0 and right.final_score == pytest.approx(1.0), label


def test_the_known_grammar_text_and_code_agree() -> None:
    """A family whose statement and implementation drift apart stops catching what it names."""

    point = 3.7e-11
    parameter = 1.2e-10
    expected = {
        "mond_simple_nu": 0.5 * point * (1.0 + math.sqrt(1.0 + 4.0 * parameter / point)),
        "mond_deep_limit": math.sqrt(point * parameter),
        "aqual_nu_form": 0.5 * point * (1.0 + math.sqrt(1.0 + 4.0 * parameter / point)),
        "mond_standard_mu": math.sqrt(
            0.5
            * (
                point**2
                + math.sqrt(point**4 + 4.0 * point**2 * parameter**2)
            )
        ),
    }
    for family in fl.MOND_AQUAL_GRAMMAR:
        assert family.evaluate(parameter, point) == pytest.approx(
            expected[family.family_id], rel=1e-12
        ), family.family_id


def test_fitting_recovers_the_planted_parameter() -> None:
    family = next(
        item for item in fl.MOND_AQUAL_GRAMMAR if item.family_id == "mond_simple_nu"
    )
    policy = fl.NoveltyPolicy()
    points = fl.RESPONSE_POINTS
    observed = [family.evaluate(1.2e-10, point) for point in points]
    parameter, distance = fl.fit_family(family, points, observed, policy)
    assert parameter == pytest.approx(1.2e-10, rel=1e-4)
    assert distance < 1e-9


def test_log_metric_is_scale_free() -> None:
    """The reason the metric is in log space: a plain relative RMS hides a small-end miss."""

    truth = [10.0**index for index in range(-6, 0)]
    wrong_at_the_small_end = [value * (10.0 if index == 0 else 1.0) for index, value in
                              enumerate(truth)]
    plain = fl.relative_rms_distance(truth, wrong_at_the_small_end)
    logged = fl.log_relative_rms_distance(truth, wrong_at_the_small_end)
    assert plain < 1e-3
    assert logged > 0.9


def test_log_metric_refuses_non_positive_values() -> None:
    assert fl.log_relative_rms_distance([1.0, -1.0], [1.0, 1.0]) == math.inf
    assert fl.log_relative_rms_distance([1.0], [0.0]) == math.inf


def test_undeclared_metric_is_refused() -> None:
    with pytest.raises(fl.FunSearchError):
        fl.NoveltyPolicy(metric="whatever")
    with pytest.raises(fl.FunSearchError):
        fl.distance_between("whatever", [1.0], [1.0])


# ---------------------------------------------------------------------------
# The corpus channel, including its equivalence-orbit test
# ---------------------------------------------------------------------------


def test_corpus_channel_zeroes_a_catalogued_pattern(corpus) -> None:
    problem = fl.declared_problems()["blinded_ratio_expansion"]
    sources = dict(fl.EXPANSION_PROBE_PROGRAMS)
    scored = fl.score_program(
        problem, sources["probe_known_corpus_pattern"], origin="probe", corpus=corpus
    )
    assert scored.quality == pytest.approx(1.0)
    assert scored.novelty.multiplier == 0.0
    assert scored.novelty.reason == "corpus_known:exact_pattern_match"
    assert scored.final_score == 0.0


def test_corpus_channel_zeroes_a_disguised_rewrite_via_the_orbit_test(corpus) -> None:
    """The disguise converges to the identical limit and is absent as a pattern."""

    problem = fl.declared_problems()["blinded_ratio_expansion"]
    sources = dict(fl.EXPANSION_PROBE_PROGRAMS)
    scored = fl.score_program(
        problem, sources["probe_known_disguised_equivalence"], origin="probe", corpus=corpus
    )
    assert scored.quality == pytest.approx(1.0)
    assert scored.novelty.reason == "corpus_known:equivalence_orbit_match"
    assert scored.novelty.multiplier == 0.0
    assert scored.novelty.detail["matched_record_id"] == "seed:euler_e_alternating"
    assert scored.novelty.detail["transformation_chain"]


def test_corpus_channel_caps_a_bare_value_match(corpus) -> None:
    problem = fl.declared_problems()["blinded_ratio_expansion"]
    sources = dict(fl.EXPANSION_PROBE_PROGRAMS)
    scored = fl.score_program(
        problem, sources["probe_unrelated_pattern"], origin="probe", corpus=corpus
    )
    assert scored.novelty.reason == "corpus_inconclusive_value_match"
    assert scored.novelty.multiplier == problem.novelty_policy.inconclusive_value_match_cap


def test_corpus_absence_is_not_novelty(corpus) -> None:
    verdict = fl.corpus_novelty(
        fl.declared_problems()["blinded_ratio_expansion"], ("41", "13", "0", "0", "0", "7", "0", "0"), corpus
    )
    assert verdict.multiplier <= 1.0
    assert "novelty claim" in json.dumps(verdict.detail) or verdict.reason != "corpus_known"
    assert fl.CLAIMS["corpus_absence_establishes_novelty"] is False


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_loop_finds_a_good_program_on_the_blinded_problem(tmp_path: Path) -> None:
    problem = fl.declared_problems()["blinded_sequence_rule"]
    config = fl.LoopConfig(generations=40, islands=4, proposals_per_call=5, seed=7)
    governor = fl.SpendGovernor(tmp_path / "ledger.json", 500, 5000, 1)
    block = fl.run_problem(
        problem, config, fl.MockMutationProposer(config.seed, problem.mutation_bank), governor
    )
    seed_quality = fl.score_program(problem, problem.seed_program).quality
    best = float(block["headline"]["best_by_quality"]["quality"])
    assert best > seed_quality + 0.2, "the loop did not improve on its own seed"
    assert block["headline"]["programs_sealed"] > 20


def test_the_loop_halts_cleanly_on_the_call_cap(tmp_path: Path) -> None:
    problem = fl.declared_problems()["blinded_sequence_rule"]
    config = fl.LoopConfig(generations=50, islands=2, proposals_per_call=2, seed=3)
    governor = fl.SpendGovernor(tmp_path / "ledger.json", 4, 5000, 1)
    block = fl.run_problem(
        problem, config, fl.MockMutationProposer(config.seed, problem.mutation_bank), governor
    )
    assert block["halt_reason"] == "call_cap_reached"
    assert governor.calls == 4
    assert block["generations_run"] == 4


def test_the_loop_halts_cleanly_on_the_dollar_cap(tmp_path: Path) -> None:
    problem = fl.declared_problems()["blinded_sequence_rule"]
    config = fl.LoopConfig(generations=50, islands=2, proposals_per_call=2, seed=3)
    governor = fl.SpendGovernor(tmp_path / "ledger.json", 500, 3, 1)
    block = fl.run_problem(
        problem, config, fl.MockMutationProposer(config.seed, problem.mutation_bank), governor
    )
    assert block["halt_reason"] == "dollar_cap_reached"
    assert governor.charged_hundredths == 3


def test_every_call_reaches_the_declared_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    fl.write_ledger(ledger, 40)
    problem = fl.declared_problems()["blinded_sequence_rule"]
    config = fl.LoopConfig(generations=5, islands=2, proposals_per_call=1, seed=3)
    governor = fl.SpendGovernor(ledger, 500, 5000, 7)
    fl.run_problem(
        problem, config, fl.MockMutationProposer(config.seed, problem.mutation_bank), governor
    )
    assert fl.read_ledger(ledger) == 40 + 5 * 7
    assert json.loads(ledger.read_text(encoding="utf-8"))["schema_version"] == fl.LEDGER_SCHEMA


def test_ledger_shape_is_enforced(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "other", "spent_dollars_hundredths": 1}), "utf-8")
    with pytest.raises(fl.FunSearchError):
        fl.read_ledger(bad)
    bad.write_text(json.dumps({"schema_version": fl.LEDGER_SCHEMA, "spent_dollars_hundredths": -1}),
                   "utf-8")
    with pytest.raises(fl.FunSearchError):
        fl.read_ledger(bad)


def test_a_broken_proposer_does_not_take_the_loop_down(tmp_path: Path) -> None:
    class Exploding:
        proposer_id = "exploding"

        def propose(self, prompt, examples, count):
            return fl.ProposalCall(self.proposer_id, "0" * 64, 1, 0, False, "provider_error", "")

        def programs(self):
            return ("def rule(n):\n    return zzz\n", "not python at all {", "")

    problem = fl.declared_problems()["blinded_sequence_rule"]
    config = fl.LoopConfig(generations=3, islands=2, proposals_per_call=2, seed=3)
    governor = fl.SpendGovernor(tmp_path / "ledger.json", 500, 5000, 1)
    block = fl.run_problem(problem, config, Exploding(), governor)
    assert block["generations_run"] == 3
    assert block["sandbox_incidents"]


def test_the_model_is_never_asked_to_adjudicate() -> None:
    """The prompt may ask for code and nothing else."""

    problem = fl.declared_problems()["blinded_response_law"]
    seeded = fl.score_program(problem, problem.seed_program, origin="seed")
    prompt = fl.build_prompt(problem, (seeded,)).lower()
    for banned in ("is this novel", "which is better", "judge", "decide", "explain why"):
        assert banned not in prompt
    assert "only the measured score counts" in prompt


# ---------------------------------------------------------------------------
# Declarations refuse malformed input
# ---------------------------------------------------------------------------


def test_sandbox_budget_refuses_undeclarable_access() -> None:
    with pytest.raises(fl.FunSearchError):
        fl.SandboxBudget(network=True)
    with pytest.raises(fl.FunSearchError):
        fl.SandboxBudget(filesystem=True)
    with pytest.raises(fl.FunSearchError):
        fl.SandboxBudget(wall_seconds=10_000.0)
    with pytest.raises(fl.FunSearchError):
        fl.SandboxBudget(import_allowlist=("math", "abc"))


def test_problem_refuses_an_undeclared_channel_or_evaluator() -> None:
    base = fl.declared_problems()["blinded_sequence_rule"]
    with pytest.raises(fl.FunSearchError):
        fl.ProblemSpec(
            problem_id="blinded_sequence_rule",
            entry="rule",
            signature_text=base.signature_text,
            seed_program=base.seed_program,
            evaluator_id="does_not_exist",
            probe_points=base.probe_points,
            novelty_channel="known_solution_grammar",
            forbidden_vocabulary=(),
            sandbox=base.sandbox,
        )
    with pytest.raises(fl.FunSearchError):
        fl.ProblemSpec(
            problem_id="blinded_sequence_rule",
            entry="rule",
            signature_text=base.signature_text,
            seed_program=base.seed_program,
            evaluator_id=base.evaluator_id,
            probe_points=base.probe_points,
            novelty_channel="vibes",
            forbidden_vocabulary=(),
            sandbox=base.sandbox,
        )


def test_problem_refuses_a_seed_that_leaks_its_subject() -> None:
    base = fl.declared_problems()["blinded_response_law"]
    with pytest.raises(fl.FunSearchError, match="leaks"):
        fl.ProblemSpec(
            problem_id="blinded_response_law",
            entry="rule",
            signature_text=base.signature_text,
            seed_program="def rule(u):\n    gravity = 1.0\n    return u * gravity\n",
            evaluator_id=base.evaluator_id,
            probe_points=base.probe_points,
            novelty_channel="known_solution_grammar",
            forbidden_vocabulary=base.forbidden_vocabulary,
            sandbox=base.sandbox,
        )


# ---------------------------------------------------------------------------
# The receipt
# ---------------------------------------------------------------------------


def test_receipt_validates(receipt: dict) -> None:
    fl.validate_receipt(receipt)


def test_receipt_claims_are_the_declared_ones(receipt: dict) -> None:
    assert receipt["claims"] == fl.CLAIMS
    assert receipt["claims"]["corpus_absence_establishes_novelty"] is False
    assert receipt["claims"]["known_solution_reproduction_scores_zero"] is True
    assert receipt["claims"]["model_never_adjudicates"] is True
    assert receipt["claims"]["model_never_saw_target"] is True
    assert receipt["claims"]["novelty_is_a_selection_pressure_not_a_postcheck"] is True


def test_receipt_seal_detects_tamper(receipt: dict) -> None:
    tampered = json.loads(json.dumps(receipt))
    tampered["problems"][0]["headline"]["programs_sealed"] += 1
    with pytest.raises(fl.FunSearchError):
        fl.validate_receipt(tampered)


def _reseal(value: dict) -> dict:
    """Re-derive both seals, so a tamper test reaches the check it is aiming at."""

    from sigma_theory_compiler.sigma_core import canonical_sha256

    core_body = {
        key: item
        for key, item in value.items()
        if key not in {"content_sha256", "result_core_sha256", "measurement"}
    }
    value["result_core_sha256"] = canonical_sha256(core_body)
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    value["content_sha256"] = canonical_sha256(body)
    return value


def test_receipt_rejects_a_forged_score(receipt: dict) -> None:
    """A restored zero, resealed end to end, must still fail on the score arithmetic."""

    tampered = json.loads(json.dumps(receipt))
    block = next(
        item for item in tampered["problems"] if item["run_label"] == "blinded_response_law"
    )
    victim = next(
        item
        for item in block["sealed_programs"]
        if float(item["novelty"]["novelty_multiplier"]) == 0.0
        and float(item["quality"]) > 0.5
    )
    victim["final_score"] = victim["quality"]
    with pytest.raises(fl.FunSearchError, match="quality"):
        fl.validate_receipt(_reseal(tampered))


def test_receipt_rejects_a_control_that_moved_a_quality(receipt: dict) -> None:
    """If emptying the grammar changed a quality, the control would prove nothing."""

    tampered = json.loads(json.dumps(receipt))
    control = next(
        item
        for item in tampered["problems"]
        if item["run_label"] == "blinded_response_law_control"
    )
    live = {
        item["program_sha256"]
        for item in next(
            block for block in tampered["problems"] if block["run_label"] == "blinded_response_law"
        )["sealed_programs"]
    }
    victim = next(item for item in control["sealed_programs"] if item["program_sha256"] in live)
    victim["quality"] = "0.123456789"
    victim["final_score"] = "0.123456789"
    with pytest.raises(fl.FunSearchError, match="control changed a quality"):
        fl.validate_receipt(_reseal(tampered))


def test_receipt_rejects_an_undeclared_run_label(receipt: dict) -> None:
    tampered = json.loads(json.dumps(receipt))
    tampered["problems"][0]["run_label"] = "something_else"
    with pytest.raises(fl.FunSearchError):
        fl.validate_receipt(_reseal(tampered))


def test_receipt_records_every_multiplier_with_a_reason(receipt: dict) -> None:
    for block in receipt["problems"]:
        for record in block["sealed_programs"]:
            assert record["novelty"]["reason"]
            assert "detail" in record["novelty"]


def test_receipt_reports_the_live_model_attempt_honestly(receipt: dict) -> None:
    attempt = receipt["proposer"]["live_model_attempt"]
    assert set(attempt) >= {"ok", "reason"}
    used = receipt["proposer"]["used"]
    assert used in {"claude_cli_oauth", "deterministic_mock_mutator"}
    if not attempt["ok"]:
        assert used == "deterministic_mock_mutator"


def test_receipt_headline_shows_the_anti_recall_result(receipt: dict) -> None:
    summary = receipt["headline"]["anti_recall"]
    assert summary["available"] is True
    assert summary["high_quality_programs_compared"] >= 6
    assert summary["zeroed_live"] >= 6
    assert summary["restored_by_control"] == summary["zeroed_live"]
    assert summary["every_quality_identical_across_runs"] is True


def test_receipt_control_isolates_the_multiplier(receipt: dict) -> None:
    fl._validate_anti_recall_control(receipt)
    labels = [block["run_label"] for block in receipt["problems"]]
    assert "blinded_response_law" in labels
    assert "blinded_response_law_control" in labels


def test_receipt_sandbox_block_is_complete(receipt: dict) -> None:
    assert receipt["sandbox"]["all_hostile_programs_contained"] is True
    assert len(receipt["sandbox"]["hostile_suite"]) == len(fl.HOSTILE_PROGRAMS)
    assert receipt["sandbox"]["runner_sha256"]


def test_receipt_budget_is_bounded(receipt: dict) -> None:
    budget = receipt["budget"]
    assert budget["calls"] <= budget["max_calls"]
    assert budget["charged_dollars_hundredths"] <= budget["max_dollars_hundredths"]
    assert budget["closing_ledger_hundredths"] == (
        budget["opening_ledger_hundredths"] + budget["charged_dollars_hundredths"]
    )


def test_receipt_seals_the_answer_without_the_loop_seeing_it(receipt: dict) -> None:
    from sigma_theory_compiler.sigma_core import canonical_sha256

    for block in receipt["problems"]:
        revealed = block["sealed_answer_revealed_after_scoring"]
        assert block["sealed_answer_sha256"] == canonical_sha256(revealed)
        for record in block["sealed_programs"]:
            if record["origin"] == "seed":
                assert revealed not in record["source"]


# ---------------------------------------------------------------------------
# Replay: every sealed score must reproduce exactly from the sealed programs
# ---------------------------------------------------------------------------


def test_replay_reproduces_every_sealed_score_exactly(receipt: dict) -> None:
    report = fl.replay_from_receipt(receipt)
    assert report["programs_checked"] > 0
    assert report["mismatches"] == []
    assert report["identical"] is True


def test_scoring_one_program_twice_is_identical() -> None:
    problem = fl.declared_problems()["blinded_response_law"]
    source = dict(fl.RESPONSE_PROBE_PROGRAMS)["probe_known_canonical"]
    first = fl.score_program(problem, source)
    second = fl.score_program(problem, source)
    assert first.to_dict() == second.to_dict()


def test_the_mock_proposer_is_deterministic() -> None:
    problem = fl.declared_problems()["blinded_sequence_rule"]
    seeded = fl.score_program(problem, problem.seed_program, origin="seed")
    outputs = []
    for _ in range(2):
        proposer = fl.MockMutationProposer(11, problem.mutation_bank)
        proposer.propose("prompt", (seeded,), 5)
        outputs.append(proposer.programs())
    assert outputs[0] == outputs[1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sigma_theory_compiler.funsearch_loop", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_cli_validate_checked() -> None:
    if not RECEIPT.exists():
        pytest.skip("receipt not present")
    completed = _cli("--validate-checked", "--output", str(RECEIPT))
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["validated"] is True


def test_cli_hostile_suite() -> None:
    completed = _cli("--hostile-suite")
    assert completed.returncode == 0, completed.stderr
    results = json.loads(completed.stdout)
    assert all(item["contained"] for item in results)


# ---------------------------------------------------------------------------
# The proposer must not degrade silently
# ---------------------------------------------------------------------------


def test_a_dead_live_proposer_fails_closed(tmp_path: Path) -> None:
    """An unreachable live proposer raises instead of quietly running the mock.

    This is the regression that mattered: an expired OAuth session turned every campaign into
    a ten-token recombination while the receipt still looked like a discovery run.
    """
    with pytest.raises(fl.ProposerUnavailable) as caught:
        fl.run_loop(
            config=fl.LoopConfig(islands=2, island_capacity=4, generations=1),
            ledger_path=tmp_path / "ledger.json",
            proposer_kind="auto",
            claude_executable="claude-executable-that-does-not-exist",
            include_corpus_problem=False,
        )
    assert "unreachable" in str(caught.value)


def test_degrading_on_purpose_is_allowed_but_marked(tmp_path: Path) -> None:
    """Opting into the mock still stamps the receipt so it cannot be read as a live run."""
    result = fl.run_loop(
        config=fl.LoopConfig(islands=2, island_capacity=4, generations=1),
        ledger_path=tmp_path / "ledger.json",
        proposer_kind="auto",
        allow_mock_fallback=True,
        claude_executable="claude-executable-that-does-not-exist",
        include_corpus_problem=False,
    )
    assert result["proposer"]["degraded"] is True
    assert result["proposer"]["used"] == "deterministic_mock_mutator"
    assert "NOT a live discovery campaign" in result["proposer"]["degraded_note"]


def test_explicit_mock_is_not_degraded_and_probes_nothing(tmp_path: Path) -> None:
    """``proposer_kind='mock'`` is a deliberate choice, not a failure, and skips the probe."""
    result = fl.run_loop(
        config=fl.LoopConfig(islands=2, island_capacity=4, generations=1),
        ledger_path=tmp_path / "ledger.json",
        proposer_kind="mock",
        claude_executable="claude-executable-that-does-not-exist",
        include_corpus_problem=False,
    )
    assert result["proposer"]["degraded"] is False
    assert result["proposer"]["live_model_attempt"]["reason"] == "not_probed"


def test_credential_loads_from_a_file_and_never_exposes_the_key(tmp_path: Path, monkeypatch) -> None:
    """A credential file populates the environment; the provenance record carries no secret."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text("# comment\nANTHROPIC_API_KEY=sk-ant-test-not-a-real-key\n", encoding="utf-8")
    record = fl.load_api_credential(root=tmp_path)
    assert record["present"] is True
    assert record["source"].endswith(".env")
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-not-a-real-key"
    serialized = json.dumps(record)
    assert "sk-ant-test-not-a-real-key" not in serialized
    assert len(record["key_sha256_prefix"]) == 12


def test_an_existing_environment_credential_is_never_overwritten(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-already-set")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n", encoding="utf-8")
    record = fl.load_api_credential(root=tmp_path)
    assert record["source"] == "environment"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-already-set"


def test_no_credential_anywhere_is_reported_not_crashed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(fl, "CREDENTIAL_FILES", (".env",))
    record = fl.load_api_credential(root=tmp_path)
    assert record == {"present": False, "source": "none", "key_sha256_prefix": ""}
