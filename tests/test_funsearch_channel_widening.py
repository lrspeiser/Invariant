"""Gates for the widened FunSearch problem: unlabelled channels the search may discover.

Before this, a declared problem was ``rule(u) -> float``.  One number in, one number out, so
a dependence on anything else was not merely hard to find -- it was unsayable, and a search
cannot discover what its grammar cannot express.  The widened problem is ``rule(u, w, x)``,
with three channels the proposer is handed positionally and told nothing about.

Three things have to be true for the widening to mean anything, and each has a control that
must fail:

*Naming is refused, not discouraged.*  ``w`` cannot leak a concept because it does not carry
one.  A declaration whose parameters are words is rejected at construction -- and the
allowlist catches a concept nobody thought to put on the forbidden list, which is exactly
where the existing word-denylist guard goes blind.

*Widening is free for anything that ignores the width.*  A program returning ``3*u`` on the
three-channel problem must produce byte-identical outputs and a bit-identical quality to the
same formula written ``rule(u)``.  The control: a program that reads ``w`` must **not** score
the same, or the extra channels are decoration.

*A channel earns its place or is refused.*  The population is adjudicated in exact integer
arithmetic by :mod:`sigma_theory_compiler.unlabelled_channel_mdl`.  The load-bearing channel
comes back certified and adopted; the third channel is present, varying and worth nothing, and
must come back refused.  A forged report that promotes it has to be rejected by the validator.

The last gate runs a real blind search with no probes seeded, so the program that reads the
second channel has to be found rather than handed over.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sigma_theory_compiler import funsearch_loop as fl

FAST_SANDBOX = fl.SandboxBudget(wall_seconds=8.0, memory_bytes=256 * 1024 * 1024)


@pytest.fixture(scope="module")
def problem() -> fl.ProblemSpec:
    return fl.declared_problems()["blinded_channel_response"]


@pytest.fixture(scope="module")
def probe_records(problem: fl.ProblemSpec) -> list[dict]:
    return [
        fl.score_program(problem, source, origin=f"probe:{label}").to_dict()
        for label, source in fl.CHANNEL_PROBE_PROGRAMS
    ]


# ---------------------------------------------------------------------------
# The naming discipline
# ---------------------------------------------------------------------------


def _respec(problem: fl.ProblemSpec, **overrides) -> fl.ProblemSpec:
    fields = {
        "problem_id": problem.problem_id,
        "entry": problem.entry,
        "signature_text": problem.signature_text,
        "seed_program": problem.seed_program,
        "evaluator_id": problem.evaluator_id,
        "probe_points": problem.probe_points,
        "novelty_channel": problem.novelty_channel,
        "forbidden_vocabulary": problem.forbidden_vocabulary,
        "sandbox": problem.sandbox,
        "channel_points": problem.channel_points,
    }
    fields.update(overrides)
    return fl.ProblemSpec(**fields)


def test_a_declared_problem_may_not_name_its_channels(problem: fl.ProblemSpec) -> None:
    """The control that must fail: naming a channel hands the proposer the concept."""

    for signature in (
        "rule(u: float, density: float, radius: float) -> float",
        "rule(u: float, w: float, temperature: float) -> float",
        "rule(obliquity: float, w: float, x: float) -> float",
    ):
        with pytest.raises(fl.FunSearchError):
            _respec(problem, signature_text=signature)


def test_the_placeholder_allowlist_sees_what_the_denylist_misses() -> None:
    assert fl.channel_name_violations("rule(u: float, w: float, x: float)", "rule") == []
    assert fl.channel_name_violations("rule(u, w, x, c3, c4)", "rule") == []
    assert fl.channel_name_violations("rule(u, gradient)", "rule") == ["gradient"]
    # A denylist only catches terms someone thought of in advance.  An allowlist on the
    # channel names catches the concept nobody listed, which is the realistic leak.
    assert fl.vocabulary_violations("rule(u, obliquity)", fl.FORBIDDEN_CHANNEL_VOCABULARY) == []
    assert fl.channel_name_violations("rule(u, obliquity)", "rule") == ["obliquity"]


def test_every_declared_problem_uses_placeholder_parameter_names() -> None:
    for declared in fl.declared_problems().values():
        assert fl.channel_name_violations(declared.signature_text, declared.entry) == []


def test_the_widened_prompt_carries_no_concept(problem: fl.ProblemSpec) -> None:
    seeded = fl.score_program(problem, problem.seed_program, origin="seed")
    prompt = fl.build_prompt(problem, (seeded,))
    assert fl.vocabulary_violations(prompt, problem.forbidden_vocabulary) == []
    assert "w" in problem.signature_text and "x" in problem.signature_text


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def test_the_widened_problem_hands_three_unlabelled_channels_to_the_entry_point(
    problem: fl.ProblemSpec,
) -> None:
    assert problem.channel_count == 3
    assert fl.signature_parameter_names(problem.signature_text, problem.entry) == ("u", "w", "x")
    rows = problem.sandbox_inputs()
    assert len(rows) == 160 == math.prod(len(levels) for levels in fl.CHANNEL_LEVELS)
    assert all(len(row) == 3 for row in rows)
    assert len(set(rows)) == len(rows)
    for index, levels in enumerate(fl.CHANNEL_LEVELS):
        assert sorted({row[index] for row in rows}) == sorted(levels)


def test_a_problem_may_not_declare_both_input_shapes(problem: fl.ProblemSpec) -> None:
    with pytest.raises(fl.FunSearchError):
        _respec(problem, probe_points=(1.0, 2.0))


def test_a_channel_table_that_disagrees_with_the_signature_is_refused(
    problem: fl.ProblemSpec,
) -> None:
    with pytest.raises(fl.FunSearchError):
        _respec(
            problem,
            signature_text="rule(u: float, w: float) -> float",
            seed_program="def rule(u, w):\n    return u",
        )
    with pytest.raises(fl.FunSearchError):
        _respec(problem, channel_points=((1.0, 2.0, 3.0), (1.0, 2.0)))
    with pytest.raises(fl.FunSearchError):
        _respec(problem, channel_points=((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)))


def test_the_original_one_number_shape_still_works() -> None:
    narrow = fl.declared_problems()["blinded_response_law"]
    assert narrow.channel_count == 1
    assert narrow.channel_points == ()
    assert narrow.sandbox_inputs() == tuple((point,) for point in narrow.probe_points)


# ---------------------------------------------------------------------------
# Widening is free for a formula that ignores the width
# ---------------------------------------------------------------------------


def test_a_program_ignoring_the_extra_channels_scores_as_a_one_variable_program(
    problem: fl.ProblemSpec,
) -> None:
    wide_source = "def rule(u, w, x):\n    return 3.0 * u\n"
    narrow_source = "def rule(u):\n    return 3.0 * u\n"
    wide = fl.run_in_sandbox(wide_source, "rule", problem.sandbox_inputs(), FAST_SANDBOX)
    narrow_rows = tuple((row[0],) for row in problem.channel_points)
    narrow = fl.run_in_sandbox(narrow_source, "rule", narrow_rows, FAST_SANDBOX)
    assert wide.ok and narrow.ok, (wide.reason, narrow.reason)
    assert wide.outputs == narrow.outputs
    measured_wide = fl.EVALUATORS[problem.evaluator_id](problem, wide.outputs)
    measured_narrow = fl.EVALUATORS[problem.evaluator_id](problem, narrow.outputs)
    assert measured_wide["quality"] == measured_narrow["quality"]
    assert fl.score_program(problem, wide_source).quality == measured_narrow["quality"]


def test_control_a_program_reading_the_second_channel_does_not_score_the_same(
    problem: fl.ProblemSpec,
) -> None:
    """The control that must fail: identical scores here would make the widening inert."""

    ignoring = fl.score_program(problem, "def rule(u, w, x):\n    return 3.0 * u\n")
    reading = fl.score_program(problem, "def rule(u, w, x):\n    return u * (2.0 + w)\n")
    third = fl.score_program(problem, "def rule(u, w, x):\n    return u * (2.0 + x)\n")
    assert ignoring.outputs != reading.outputs
    assert reading.quality > ignoring.quality
    assert reading.quality == 1.0
    # The third channel is present and varying and worth less than ignoring it entirely.
    assert third.quality < ignoring.quality


def test_writing_a_channel_down_and_cancelling_it_changes_no_score(
    problem: fl.ProblemSpec,
) -> None:
    plain = fl.score_program(problem, "def rule(u, w, x):\n    return 3.0 * u\n")
    inert = fl.score_program(problem, "def rule(u, w, x):\n    return 3.0 * u + 0.0 * w\n")
    assert plain.outputs == inert.outputs
    assert plain.quality == inert.quality


def test_the_mutation_bank_offers_every_channel_on_equal_terms(
    problem: fl.ProblemSpec,
) -> None:
    assert {"u", "w", "x"} <= set(problem.mutation_bank)
    with_second = sum(1 for item in problem.mutation_bank if "w" in item)
    with_third = sum(1 for item in problem.mutation_bank if "x" in item)
    assert with_second == with_third


# ---------------------------------------------------------------------------
# The two lanes are the same numbers
# ---------------------------------------------------------------------------


def test_the_search_lane_and_the_certificate_lane_agree_row_for_row(
    problem: fl.ProblemSpec,
) -> None:
    table = fl.sealed_channel_table()
    assert table.arity == problem.channel_count
    assert len(table.rows) == len(problem.channel_points)
    for exact_row, float_row in zip(table.rows, problem.channel_points, strict=True):
        for exact, approximate in zip(exact_row, float_row, strict=True):
            assert float(exact) == approximate


# ---------------------------------------------------------------------------
# Adjudication: a channel earns its place or is refused
# ---------------------------------------------------------------------------


def test_the_discovery_report_names_the_load_bearing_channel(
    problem: fl.ProblemSpec, probe_records
) -> None:
    report = fl.channel_discovery_report(problem, probe_records)
    assert report["adjudicated"] is True
    assert report["load_bearing_channels"] == ["w"]
    assert "x" in report["not_load_bearing_channels"]
    second = report["verdicts"][1]
    assert "no function of u alone" in second["headline"]
    assert "channel w is load bearing" in second["headline"]
    assert second["adoption"]["net_bits"] >= second["adoption"]["admission_bits"]
    assert (
        second["obstruction"]["exhibited_data_bits_using_the_channel"]
        < second["obstruction"]["floor_data_bits"]
    )
    fl._validate_channel_discovery(report)


def test_control_the_third_channel_is_refused(problem: fl.ProblemSpec, probe_records) -> None:
    report = fl.channel_discovery_report(problem, probe_records)
    third = report["verdicts"][2]
    assert third["channel_symbol"] == "x"
    assert third["channel_status"] == "NOT_LOAD_BEARING"
    assert third["adoption"]["verdict"] == "NOT_ADOPTED"
    assert third["obstruction"]["verdict"] == "NOT_CERTIFIED"
    assert third["obstruction"]["floor_data_bits"] == (
        third["obstruction"]["trivial_floor_data_bits"]
    )


def test_the_discovery_report_never_says_what_a_channel_is(
    problem: fl.ProblemSpec, probe_records
) -> None:
    report = fl.channel_discovery_report(problem, probe_records)
    rendered = json.dumps(report, sort_keys=True)
    assert fl.vocabulary_violations(rendered, problem.forbidden_vocabulary) == []
    assert fl.vocabulary_violations(rendered, ("obliquity", "density", "radius")) == []


def test_a_population_that_never_read_the_channel_gets_no_verdict(
    problem: fl.ProblemSpec,
) -> None:
    records = [
        fl.score_program(problem, source).to_dict()
        for source in (
            "def rule(u, w, x):\n    return 3.0 * u\n",
            "def rule(u, w, x):\n    return 2.0 * u\n",
        )
    ]
    report = fl.channel_discovery_report(problem, records)
    assert report["load_bearing_channels"] == []
    assert "w" in report["undetermined_channels"]
    fl._validate_channel_discovery(report)


def test_the_validator_refuses_an_unpaid_load_bearing_claim(
    problem: fl.ProblemSpec, probe_records
) -> None:
    report = fl.channel_discovery_report(problem, probe_records)
    fl._validate_channel_discovery(report)
    promoted = json.loads(json.dumps(report))
    promoted["verdicts"][2]["channel_status"] = "LOAD_BEARING"
    promoted["load_bearing_channels"] = ["w", "x"]
    with pytest.raises(fl.FunSearchError):
        fl._validate_channel_discovery(promoted)
    understated = json.loads(json.dumps(report))
    understated["verdicts"][1]["adoption"]["net_bits"] = 1
    with pytest.raises(fl.FunSearchError):
        fl._validate_channel_discovery(understated)
    mismatched = json.loads(json.dumps(report))
    mismatched["load_bearing_channels"] = ["u", "w", "x"]
    with pytest.raises(fl.FunSearchError):
        fl._validate_channel_discovery(mismatched)


# ---------------------------------------------------------------------------
# The search itself
# ---------------------------------------------------------------------------


def test_a_blind_search_discovers_the_load_bearing_channel(
    problem: fl.ProblemSpec, tmp_path: Path
) -> None:
    """No probes seeded: the mutator has to reach for the second channel by itself."""

    config = fl.LoopConfig(
        generations=25, islands=3, proposals_per_call=4, island_capacity=8, seed=11
    )
    governor = fl.SpendGovernor(tmp_path / "ledger.json", 5000, 100000, 0)
    block = fl.run_problem(
        problem, config, fl.MockMutationProposer(11, problem.mutation_bank), governor
    )
    best = max(block["sealed_programs"], key=lambda record: float(record["quality"]))
    assert float(best["quality"]) == 1.0
    assert best["origin"] == "proposed"
    assert "w" in best["source"]
    report = fl.channel_discovery_report(problem, block["sealed_programs"])
    assert report["load_bearing_channels"] == ["u", "w"]
    assert report["not_load_bearing_channels"] == ["x"]
    fl._validate_channel_discovery(report)
