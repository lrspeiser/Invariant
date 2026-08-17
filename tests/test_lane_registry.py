"""A5 lane-registry gates.

The registry's whole value is that an unattempted lane is a *recorded fact*, so the
load-bearing tests are coverage and typing: every declared problem reaches at least one
lane, every declared lane is either reachable or explicitly marked as awaiting a problem
kind, every skip carries a code from the frozen vocabulary, and every code in that
vocabulary is actually produced by some (problem, lane) pair — an unreachable reason is
as dishonest as an untyped one.  The pin test makes adding a lane a deliberate act.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.dozen_unsolved_progress_campaign import DOZEN_IDS
from sigma_theory_compiler.lane_registry import (
    EFFECTIVE_ROW_CAP,
    LANE_IDS,
    LANES,
    REGISTRY_CONTENT_SHA256,
    RESOURCES,
    SKIP_REASONS,
    UNSOLVED_DOZEN_ROSTER,
    LaneRegistryError,
    applicable_lanes,
    declared_blocker_vocabulary,
    declared_row_budget,
    fanout_plan,
    lane,
    lane_decisions,
    lane_for_stage,
    lanes_by_resource,
    lanes_emitting,
    main,
    registry_declaration,
    render_table,
    skipped_lanes,
)
from sigma_theory_compiler.problem_queue import load_queue

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_V3 = REPO_ROOT / "configs" / "problem_queue_v3.json"


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    return list(load_queue(QUEUE_V3)["entries"])


# ---------------------------------------------------------------------------
# The frozen declaration
# ---------------------------------------------------------------------------


def test_registry_lane_id_set_is_pinned() -> None:
    """Adding, removing, or renaming a lane must be a deliberate, reviewed edit."""

    assert LANE_IDS == (
        "row_generation",
        "conjecture_generation",
        "basis_synthesis",
        "nonlinear_coefficient_search",
        "structural_repair",
        "holonomic_guesser",
        "spectral_signal_scan",
        "lemma_decomposition",
        "quantified_inequality_proofs",
        "gpu_counterexample_sweep",
        "exponent_diophantine_sweeper",
        "sat_certificate_lane",
        "inverse_symbolic_engine",
        "dozen_unsolved_progress_campaign",
        "gpu_campaign_receipt_binding",
    )


def test_registry_content_hash_is_pinned() -> None:
    assert REGISTRY_CONTENT_SHA256 == (
        "78b06b85ff4bcd2169825ebe408e926e53b4b9d993b67cc5be58ce18f5eff827"
    )
    declaration = registry_declaration()
    assert declaration["content_sha256"] == REGISTRY_CONTENT_SHA256
    assert declaration["claims"]["scalar_truth_or_probability_score"] is False
    assert declaration["claims"]["skips_are_recorded_never_silent"] is True


def test_registry_is_internally_wellformed() -> None:
    stages = [item.stage for item in LANES]
    assert len(set(stages)) == len(stages), "each lane owns exactly one stage name"
    assert len(set(LANE_IDS)) == len(LANE_IDS)
    for spec in LANES:
        assert spec.resource in RESOURCES
        assert spec.typical_cost in ("fast", "medium", "slow")
        assert spec.module.startswith("sigma_theory_compiler.")
        assert spec.entry_point
        assert spec.preconditions or spec.stage == "generate_rows" or spec.accepts_kinds
        assert lane_for_stage(spec.stage) is spec
        assert lane(spec.lane_id) is spec


def test_unknown_lane_and_resource_fail_closed() -> None:
    with pytest.raises(LaneRegistryError, match="unknown lane"):
        lane("no_such_lane")
    with pytest.raises(LaneRegistryError, match="unknown resource"):
        lanes_by_resource("tpu")


def test_row_cap_and_rosters_match_their_sources() -> None:
    """The registry restates other modules' constants; drift must fail here, loudly."""

    from sigma_theory_compiler.discovery_scheduler import GENERATOR_CAPS

    assert EFFECTIVE_ROW_CAP == GENERATOR_CAPS["max_rows"]
    assert UNSOLVED_DOZEN_ROSTER == tuple(sorted(DOZEN_IDS))


# ---------------------------------------------------------------------------
# Coverage over the sealed queue
# ---------------------------------------------------------------------------


def test_every_queue_problem_reaches_at_least_one_lane(entries: list[dict]) -> None:
    for entry in entries:
        assert applicable_lanes(entry), f"{entry['id']} has no applicable lane"


def test_every_lane_is_reachable_or_declared_awaiting_a_problem_kind(
    entries: list[dict],
) -> None:
    reached = {
        spec.lane_id for entry in entries for spec in applicable_lanes(entry)
    }
    for spec in LANES:
        if spec.lane_id in reached:
            assert not spec.awaiting_problem_kind, (
                f"{spec.lane_id} is reachable but claims to be awaiting a problem kind"
            )
        else:
            assert spec.awaiting_problem_kind, (
                f"{spec.lane_id} is unreachable from the queue and is not declared "
                "awaiting_problem_kind"
            )
    assert {spec.lane_id for spec in LANES if spec.awaiting_problem_kind} == {
        "inverse_symbolic_engine",
        "sat_certificate_lane",
    }


def test_every_lane_decision_is_returned_for_every_lane(entries: list[dict]) -> None:
    """Nothing is dropped: applicable + skipped is always the whole registry."""

    for entry in entries:
        decisions = lane_decisions(entry)
        assert [item.lane_id for item in decisions] == list(LANE_IDS)
        assert len(applicable_lanes(entry)) + len(skipped_lanes(entry)) == len(LANES)


def test_skip_reasons_are_typed_and_exhaustive(entries: list[dict]) -> None:
    seen: set[str] = set()
    for entry in entries:
        for decision in skipped_lanes(entry):
            assert decision.skip_reason in SKIP_REASONS, decision.skip_reason
            assert decision.detail, "a skip must say why in prose too"
            seen.add(decision.skip_reason)
    assert seen == set(SKIP_REASONS), (
        f"declared but never produced: {sorted(set(SKIP_REASONS) - seen)}"
    )


def test_named_skip_cases_are_the_real_ones(entries: list[dict]) -> None:
    """Spot-check the three skips a reader would want to verify by hand."""

    by_id = {entry["id"]: entry for entry in entries}

    twin = {item.lane_id: item for item in skipped_lanes(by_id["twin_prime_infinitude"])}
    assert twin["spectral_signal_scan"].skip_reason == "insufficient_rows"
    assert twin["holonomic_guesser"].skip_reason == "insufficient_rows"
    assert declared_row_budget(by_id["twin_prime_infinitude"]) == 7

    brocard = {item.lane_id: item for item in skipped_lanes(by_id["brocard_problem"])}
    assert brocard["exponent_diophantine_sweeper"].skip_reason == "equation_not_in_sweeper_roster"
    assert brocard["row_generation"].skip_reason == "kind_mismatch"

    collatz = {item.lane_id: item for item in skipped_lanes(by_id["collatz_stopping_time"])}
    assert collatz["dozen_unsolved_progress_campaign"].skip_reason == "not_in_declared_roster"
    assert collatz["inverse_symbolic_engine"].skip_reason == "needs_target_constant"
    assert collatz["sat_certificate_lane"].skip_reason == "needs_bounded_coloring_statement"


def test_row_budget_is_capped_by_the_host_generators(entries: list[dict]) -> None:
    by_id = {entry["id"]: entry for entry in entries}
    # The queue asks for two million Pascal rows; the host caps every generator at 64.
    assert by_id["singmaster_conjecture"]["machine_form"]["max_point"] == 1_000_000
    assert declared_row_budget(by_id["singmaster_conjecture"]) == EFFECTIVE_ROW_CAP
    assert declared_row_budget(by_id["erdos_straus"]) is None


def test_physics_and_module_targets_are_not_laneless(entries: list[dict]) -> None:
    by_id = {entry["id"]: entry for entry in entries}
    for physics_id in ("baryonic_rotation_law", "cluster_missing_mass"):
        assert [spec.lane_id for spec in applicable_lanes(by_id[physics_id])] == [
            "gpu_campaign_receipt_binding"
        ]
    assert [spec.lane_id for spec in applicable_lanes(by_id["quantified_inequality_families"])] == [
        "lemma_decomposition",
        "quantified_inequality_proofs",
    ]


# ---------------------------------------------------------------------------
# The declared blocker vocabulary
# ---------------------------------------------------------------------------


def test_declared_blocker_vocabulary_covers_the_known_typed_blockers() -> None:
    assert lanes_emitting("missing_prover:sign") == ("quantified_inequality_proofs",)
    assert lanes_emitting("missing_prover:closed_form") == ("lemma_decomposition",)
    assert lanes_emitting("missing_sweeper:diophantine_family") == ("gpu_counterexample_sweep",)
    # A wildcard declaration covers every subject of its kind.
    assert lanes_emitting("missing_generator:sealed_catalan_like_recurrence_v1") == (
        "row_generation",
    )
    assert lanes_emitting("statement_kinds_too_weak") == ("dozen_unsolved_progress_campaign",)
    # A blocker no lane declares is honestly reported as unowned, not guessed at.
    assert lanes_emitting("missing_adapter:uv_form_factor_operator") == ()
    vocabulary = declared_blocker_vocabulary()
    assert "missing_prover:sign" in vocabulary
    assert "CAP_TRIPPED:max_vars" in vocabulary


def test_dozen_blocker_vocabulary_matches_the_campaign_table() -> None:
    from sigma_theory_compiler.dozen_unsolved_progress_campaign import FIRST_BLOCKERS

    declared = set(lane("dozen_unsolved_progress_campaign").emits_blockers)
    assert declared == {item["code"] for item in FIRST_BLOCKERS.values()}


# ---------------------------------------------------------------------------
# Plan and CLI
# ---------------------------------------------------------------------------


def test_fanout_plan_counts_are_the_sum_of_the_decisions(entries: list[dict]) -> None:
    plan = fanout_plan(entries)
    assert plan["counts"]["problems"] == 25
    assert plan["counts"]["lanes"] == len(LANES)
    assert plan["counts"]["attempts_planned"] == sum(
        len(applicable_lanes(entry)) for entry in entries
    )
    assert (
        plan["counts"]["attempts_planned"] + plan["counts"]["skips_recorded"]
        == 25 * len(LANES)
    )
    assert plan["registry_content_sha256"] == REGISTRY_CONTENT_SHA256


def test_render_table_is_deterministic_and_lists_every_lane() -> None:
    first = render_table()
    assert first == render_table()
    for spec in LANES:
        assert f"`{spec.lane_id}`" in first


def test_cli_validate_and_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--validate-checked"]) == 0
    assert REGISTRY_CONTENT_SHA256 in capsys.readouterr().out
    assert main(["--table"]) == 0
    assert "| lane_id |" in capsys.readouterr().out
    assert main(["--queue", str(QUEUE_V3)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["counts"]["problems"] == 25
