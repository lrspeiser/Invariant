"""Gates for the multi-model ensemble proposer.

The module exists to answer one question the engine could not previously ask: *did the
expensive model earn its cost?*  So the tests are organised around the three things that have
to be true before that answer means anything, and every one of them is paired with a control
that must fail.

*The declared weights are what actually ran.*  The draw schedule is a largest-remainder
allocation over integers, so the realized allocation is checkable by cross-multiplication with
no floating point.  Control: a receipt whose recorded allocation does not honour its own
declared weights is rejected.

*The attribution is real.*  Every proposed program names the model that returned it, and the
per-model counts must partition the population.  Controls: an attribution naming an undeclared
model, a first-seal count that does not add up, and a model that returned programs while being
charged nothing are each rejected.

*The measure can say no.*  Two models whose behaviour is identical must both score
``contributed_nothing``, because neither one contributes anything the other does not.  This
is the load-bearing control: a yield measure that cannot return zero is decoration, and the
behaviour clustering is deliberately generous, so a zero here is a zero that was hard to get.

Nothing in this file spends anything.  Every model is a deterministic stand-in -- a scripted
proposer with a fixed program list, or the seeded mutator from the FunSearch loop -- so a
two-model ensemble is proved to attribute correctly without a single provider call.
"""

from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler import funsearch_loop as fl
from sigma_theory_compiler import model_ensemble_proposer as me

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "runs" / "math" / "model-ensemble" / "ensemble-v1.json"


def _program(value: str) -> str:
    return f"def rule(n):\n    return {value}\n"


class ScriptedProposer:
    """A model that always returns the same declared programs, whatever it is asked.

    Deterministic to the byte, which is what lets a test assert exact attribution arithmetic
    rather than "roughly the right shape".
    """

    def __init__(self, sources, proposer_id: str = "scripted") -> None:
        self.proposer_id = proposer_id
        self._sources = tuple(sources)
        self.prompts: list[str] = []
        self.counts: list[int] = []

    def propose(self, prompt, examples, count):
        self.prompts.append(prompt)
        self.counts.append(count)
        return fl.ProposalCall(self.proposer_id, "0" * 64, len(prompt), len(self._sources),
                               True, "returned_programs", "")

    def programs(self):
        return self._sources


def _two_model(alpha_sources, beta_sources, **kwargs):
    ensemble = me.ModelEnsemble(
        slots=(
            me.ModelSlot(name="alpha", weight=kwargs.get("alpha_weight", 1),
                         cost_units=kwargs.get("alpha_cost", 1)),
            me.ModelSlot(name="beta", weight=kwargs.get("beta_weight", 1),
                         cost_units=kwargs.get("beta_cost", 1)),
        )
    )
    delegates = {
        "alpha": ScriptedProposer(alpha_sources, "alpha"),
        "beta": ScriptedProposer(beta_sources, "beta"),
    }
    return ensemble, delegates


def _run(ensemble, delegates, tmp_path: Path, *, generations: int = 4, islands: int = 1):
    problem = fl.declared_problems()["blinded_sequence_rule"]
    config = fl.LoopConfig(
        generations=generations, islands=islands, proposals_per_call=3, reset_period=0, seed=11
    )
    governor = fl.SpendGovernor(tmp_path / "ledger.json", 500, 5000, 1)
    proposer = me.EnsembleProposer(ensemble, delegates, max_calls=generations * islands + 8)
    block = me.run_ensemble_problem(problem, config, proposer, governor)
    block["run_label"] = "ensemble"
    return block


# ---------------------------------------------------------------------------
# The declaration: integer weights, exact schedule
# ---------------------------------------------------------------------------


def test_a_full_cycle_realizes_the_declared_weights_exactly() -> None:
    ensemble = me.ModelEnsemble(
        slots=(
            me.ModelSlot(name="a", weight=3),
            me.ModelSlot(name="b", weight=1),
            me.ModelSlot(name="c", weight=2),
        )
    )
    cycle = ensemble.cycle()
    assert len(cycle) == 6
    assert cycle.count("a") == 3
    assert cycle.count("b") == 1
    assert cycle.count("c") == 2


def test_the_schedule_never_drifts_more_than_one_call_from_its_weights() -> None:
    """The largest-remainder bound: ``|c_i * W - w_i * k| < W`` after any ``k`` draws."""

    ensemble = me.ModelEnsemble(
        slots=(
            me.ModelSlot(name="a", weight=7),
            me.ModelSlot(name="b", weight=2),
            me.ModelSlot(name="c", weight=1),
        )
    )
    total = ensemble.total_weight
    for calls in range(1, 61):
        drawn = ensemble.schedule(calls)
        counts = {name: drawn.count(name) for name in ensemble.names}
        for slot in ensemble.slots:
            assert abs(counts[slot.name] * total - slot.weight * calls) < total
        assert ensemble.allocation_is_exact(counts)


def test_an_allocation_that_ignores_the_weights_is_refused() -> None:
    """The control for the test above: the bound has teeth."""

    ensemble = me.ModelEnsemble(
        slots=(me.ModelSlot(name="a", weight=9), me.ModelSlot(name="b", weight=1))
    )
    assert not ensemble.allocation_is_exact({"a": 5, "b": 5})
    assert ensemble.allocation_is_exact({"a": 9, "b": 1})


def test_weights_must_be_integers_so_the_schedule_stays_exact() -> None:
    with pytest.raises(me.EnsembleError):
        me.ModelSlot(name="a", weight=0)
    with pytest.raises(me.EnsembleError):
        me.ModelSlot(name="a", weight=2, cost_units=0)
    with pytest.raises(me.EnsembleError):
        me.ModelSlot(name="", weight=1)


def test_duplicate_model_names_are_refused() -> None:
    """Two slots with one name would make attribution meaningless before it started."""

    with pytest.raises(me.EnsembleError):
        me.ModelEnsemble(
            slots=(me.ModelSlot(name="a", weight=1), me.ModelSlot(name="a", weight=2))
        )


def test_an_undeclared_draw_policy_is_refused() -> None:
    with pytest.raises(me.EnsembleError):
        me.ModelEnsemble(slots=(me.ModelSlot(name="a", weight=1),), draw_policy="vibes")


def test_a_total_weight_beyond_the_declared_maximum_is_refused() -> None:
    """The cycle is carried whole in the receipt, so it has to stay readable."""

    me.ModelEnsemble(
        slots=(
            me.ModelSlot(name="a", weight=me.MAX_TOTAL_WEIGHT - 1),
            me.ModelSlot(name="b", weight=1),
        )
    )
    with pytest.raises(me.EnsembleError):
        me.ModelEnsemble(
            slots=(
                me.ModelSlot(name="a", weight=me.MAX_TOTAL_WEIGHT),
                me.ModelSlot(name="b", weight=1),
            )
        )


def test_the_weighted_random_policy_is_seeded_and_reproducible() -> None:
    spec = me.ModelEnsemble(
        slots=(me.ModelSlot(name="a", weight=3), me.ModelSlot(name="b", weight=1)),
        draw_policy="weighted_random",
        seed=4242,
    )
    assert spec.schedule(50) == spec.schedule(50)
    assert spec.schedule(50) != me.ModelEnsemble(
        slots=spec.slots, draw_policy="weighted_random", seed=99
    ).schedule(50)


# ---------------------------------------------------------------------------
# The proposer: routing, delegates, prompt pass-through
# ---------------------------------------------------------------------------


def test_calls_are_routed_to_the_declared_models_in_the_declared_proportion() -> None:
    ensemble, delegates = _two_model((_program("1"),), (_program("2"),),
                                     alpha_weight=3, beta_weight=1)
    proposer = me.EnsembleProposer(ensemble, delegates, max_calls=40)
    for _ in range(8):
        proposer.propose("prompt", (), 3)
    assert proposer.calls_by_model == {"alpha": 6, "beta": 2}


def test_a_missing_or_extra_delegate_is_refused() -> None:
    ensemble, delegates = _two_model((_program("1"),), (_program("2"),))
    with pytest.raises(me.EnsembleError):
        me.EnsembleProposer(ensemble, {"alpha": delegates["alpha"]})
    with pytest.raises(me.EnsembleError):
        me.EnsembleProposer(
            ensemble, {**delegates, "gamma": ScriptedProposer((), "gamma")}
        )


def test_the_prompt_reaches_every_model_byte_identically() -> None:
    """Blindness (I1) survives the ensemble only if there is one prompt, not one per model."""

    ensemble, delegates = _two_model((_program("1"),), (_program("2"),))
    proposer = me.EnsembleProposer(ensemble, delegates, max_calls=40)
    prompt = "the exact bytes handed in"
    for _ in range(6):
        proposer.propose(prompt, (), 3)
    seen = delegates["alpha"].prompts + delegates["beta"].prompts
    assert seen and all(item == prompt for item in seen)
    assert all(name not in prompt for name in ensemble.names)


def test_a_proposer_that_rewrites_the_prompt_would_be_caught() -> None:
    """The control for the test above: the equality check is not vacuous."""

    class Leaky(ScriptedProposer):
        def propose(self, prompt, examples, count):
            return super().propose(prompt + f"\n# you are {self.proposer_id}", examples, count)

    ensemble, _ = _two_model((), ())
    delegates = {"alpha": Leaky((_program("1"),), "alpha"),
                 "beta": Leaky((_program("2"),), "beta")}
    proposer = me.EnsembleProposer(ensemble, delegates, max_calls=40)
    proposer.propose("clean", (), 3)
    proposer.propose("clean", (), 3)
    seen = delegates["alpha"].prompts + delegates["beta"].prompts
    assert not all(item == "clean" for item in seen)


def test_a_slot_proposal_count_overrides_the_loop_setting() -> None:
    """This is how "the fast model explores breadth" is declared rather than asserted."""

    ensemble = me.ModelEnsemble(
        slots=(
            me.ModelSlot(name="wide", weight=1, proposals_per_call=9),
            me.ModelSlot(name="narrow", weight=1, proposals_per_call=2),
        )
    )
    delegates = {"wide": ScriptedProposer((), "wide"), "narrow": ScriptedProposer((), "narrow")}
    proposer = me.EnsembleProposer(ensemble, delegates, max_calls=10)
    proposer.propose("p", (), 4)
    proposer.propose("p", (), 4)
    assert delegates["wide"].counts == [9]
    assert delegates["narrow"].counts == [2]


def test_the_claude_ensemble_pins_one_model_per_slot() -> None:
    """Construction only -- nothing is invoked, so this test spends nothing."""

    ensemble = me.ModelEnsemble(
        slots=(me.ModelSlot(name="haiku", weight=3), me.ModelSlot(name="opus", weight=1))
    )
    delegates = me.build_claude_ensemble(ensemble, executable="claude")
    assert sorted(delegates) == ["haiku", "opus"]
    assert delegates["haiku"]._model == "haiku"
    assert delegates["opus"]._model == "opus"


def test_the_schedule_cannot_be_consumed_past_its_declared_length() -> None:
    ensemble, delegates = _two_model((), ())
    proposer = me.EnsembleProposer(ensemble, delegates, max_calls=2)
    proposer.propose("p", (), 1)
    proposer.propose("p", (), 1)
    with pytest.raises(me.EnsembleError):
        proposer.propose("p", (), 1)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def test_every_proposed_program_names_the_model_that_returned_it(tmp_path: Path) -> None:
    ensemble, delegates = _two_model(
        (_program("n"), _program("n * 2")), (_program("n * n"), _program("n + 3"))
    )
    block = _run(ensemble, delegates, tmp_path, generations=6)
    proposed = [r for r in block["sealed_programs"] if r["origin"] == "proposed"]
    assert proposed
    for record in proposed:
        assert record["attribution"]["proposed_by"] in {"alpha", "beta"}
    me.validate_block(block)


def test_disjoint_models_are_attributed_disjointly(tmp_path: Path) -> None:
    ensemble, delegates = _two_model(
        (_program("n"), _program("n * 2")), (_program("n * n"), _program("n + 3"))
    )
    block = _run(ensemble, delegates, tmp_path, generations=6)
    by_model: dict[str, set[str]] = {"alpha": set(), "beta": set()}
    for record in block["sealed_programs"]:
        for name in record["attribution"]["times_returned_by_model"]:
            by_model[name].add(record["source"])
    assert by_model["alpha"] == {_program("n"), _program("n * 2")}
    assert by_model["beta"] == {_program("n * n"), _program("n + 3")}
    assert not by_model["alpha"] & by_model["beta"]


def test_a_program_both_models_returned_credits_both(tmp_path: Path) -> None:
    """Attribution is by return: the shared program is not the property of whoever was first."""

    shared = _program("n * n + 1")
    ensemble, delegates = _two_model((shared, _program("n")), (shared, _program("n + 5")))
    block = _run(ensemble, delegates, tmp_path, generations=6)
    record = next(r for r in block["sealed_programs"] if r["source"] == shared)
    attribution = record["attribution"]
    assert set(attribution["times_returned_by_model"]) == {"alpha", "beta"}
    assert attribution["proposed_by"] == "alpha"
    assert attribution["also_proposed_by"] == ["beta"]
    assert attribution["times_returned"] == sum(
        attribution["times_returned_by_model"].values()
    )


def test_founders_carry_no_attribution(tmp_path: Path) -> None:
    ensemble, delegates = _two_model((_program("n"),), (_program("n * n"),))
    block = _run(ensemble, delegates, tmp_path, generations=4)
    seeds = [r for r in block["sealed_programs"] if r["origin"] == "seed"]
    assert seeds
    for record in seeds:
        assert record["attribution"]["proposed_by"] == ""
        assert record["attribution"]["times_returned_by_model"] == {}


# ---------------------------------------------------------------------------
# Yield and the ablation
# ---------------------------------------------------------------------------


def test_per_model_yield_reports_the_three_declared_numbers(tmp_path: Path) -> None:
    ensemble, delegates = _two_model(
        (_program("n"), _program("n * 2")), (_program("n * n"), _program("n + 3"))
    )
    block = _run(ensemble, delegates, tmp_path, generations=6)
    rows = {row["model"]: row for row in block["model_yield"]["rows"]}
    assert set(rows) == {"alpha", "beta"}
    for row in rows.values():
        assert row["programs_returned"] > 0
        assert row["distinct_behaviours_contributed"] > 0
        assert float(row["best_quality_reached"]) > 0.0
    assert sum(row["programs_sealed_first"] for row in rows.values()) == sum(
        1 for r in block["sealed_programs"] if r["attribution"]["proposed_by"]
    )


def test_two_models_with_disjoint_behaviour_each_earn_their_cost(tmp_path: Path) -> None:
    ensemble, delegates = _two_model(
        (_program("n"), _program("n * 2")), (_program("n * n"), _program("n + 3"))
    )
    block = _run(ensemble, delegates, tmp_path, generations=6)
    rows = {row["model"]: row for row in block["ablation"]["rows"]}
    assert rows["alpha"]["contribution_verdict"] == "contributed"
    assert rows["beta"]["contribution_verdict"] == "contributed"
    assert rows["alpha"]["behaviours_lost_by_removal"] > 0
    assert rows["beta"]["behaviours_lost_by_removal"] > 0
    assert block["headline"]["models_that_contributed_nothing"] == []


def test_two_models_with_identical_behaviour_earn_nothing(tmp_path: Path) -> None:
    """The control the whole module rests on: the measure must be able to say no.

    Both models return the same two programs, so neither contributes a behaviour the other
    does not, and neither can be credited with anything.  The behaviour clustering is
    deliberately generous -- it can split a behaviour but never merge two -- so a zero here
    survived a test tilted towards finding a contribution.
    """

    shared = (_program("n * n + 2"), _program("n + 7"))
    ensemble, delegates = _two_model(shared, shared)
    block = _run(ensemble, delegates, tmp_path, generations=6)
    rows = {row["model"]: row for row in block["ablation"]["rows"]}
    for name in ("alpha", "beta"):
        assert rows[name]["behaviours_lost_by_removal"] == 0
        assert rows[name]["quality_lost_by_removal"] == "0.000000000"
        assert rows[name]["contribution_verdict"] == "contributed_nothing"
        assert rows[name]["cost_verdict"] == "did_not_earn_its_cost"
    assert sorted(block["headline"]["models_that_contributed_nothing"]) == ["alpha", "beta"]
    me.validate_block(block)


def test_an_expensive_model_that_adds_little_fails_the_cost_test(tmp_path: Path) -> None:
    """Contribution and cost are different questions, and this is a case where they differ."""

    ensemble, delegates = _two_model(
        (_program("n"), _program("n * 2"), _program("n * 3"), _program("n * 5")),
        (_program("n * n"),),
        alpha_weight=1,
        beta_weight=1,
        beta_cost=40,
    )
    block = _run(ensemble, delegates, tmp_path, generations=8)
    rows = {row["model"]: row for row in block["ablation"]["rows"]}
    assert rows["beta"]["contribution_verdict"] == "contributed"
    assert not rows["beta"]["unique_share_at_least_spend_share"]
    assert rows["beta"]["cost_verdict"] == "did_not_earn_its_cost"
    assert rows["alpha"]["cost_verdict"] == "earned_its_cost"
    spend = Fraction(*(int(p) for p in rows["beta"]["spend_share"].split("/")))
    unique = Fraction(*(int(p) for p in rows["beta"]["unique_behaviour_share"].split("/")))
    assert unique < spend


def test_the_costlier_model_is_charged_more_from_the_same_governor(tmp_path: Path) -> None:
    ensemble, delegates = _two_model(
        (_program("n"),), (_program("n * n"),), alpha_cost=1, beta_cost=8
    )
    block = _run(ensemble, delegates, tmp_path, generations=8)
    charged = block["charged_hundredths_by_model"]
    assert charged["beta"] == 8 * charged["alpha"]
    assert sum(charged.values()) == sum(
        call["charged_hundredths"] for call in block["proposal_calls"]
    )
    assert fl.read_ledger(tmp_path / "ledger.json") == sum(charged.values())


def test_the_cap_binds_the_campaign_rather_than_skipping_the_expensive_model(
    tmp_path: Path,
) -> None:
    """Skipping a model that no longer fits would silently rewrite the declared weights."""

    ensemble, delegates = _two_model(
        (_program("n"),), (_program("n * n"),), alpha_cost=1, beta_cost=50
    )
    problem = fl.declared_problems()["blinded_sequence_rule"]
    config = fl.LoopConfig(generations=20, islands=1, proposals_per_call=2, reset_period=0)
    governor = fl.SpendGovernor(tmp_path / "ledger.json", 500, 20, 1)
    proposer = me.EnsembleProposer(ensemble, delegates, max_calls=40)
    block = me.run_ensemble_problem(problem, config, proposer, governor)
    assert block["halt_reason"] == "dollar_cap_reached"
    assert governor.charged_hundredths <= 20
    # The calls that were made are a PREFIX of the declared schedule: the run stopped at the
    # first call it could not afford instead of stepping over it to reach a cheaper one.
    drawn = tuple(call["model"] for call in block["proposal_calls"])
    assert drawn == ensemble.schedule(len(drawn))
    assert "beta" in ensemble.schedule(len(drawn) + 1)


class FailingProposer(ScriptedProposer):
    """Fails its first ``failures`` calls with a declared reason, then behaves."""

    def __init__(self, sources, proposer_id: str, reason: str, detail: str, failures: int):
        super().__init__(sources, proposer_id)
        self._reason = reason
        self._detail = detail
        self._left = failures

    def propose(self, prompt, examples, count):
        self.prompts.append(prompt)
        self.counts.append(count)
        if self._left > 0:
            self._left -= 1
            self._sources_backup = self._sources
            return fl.ProposalCall(
                self.proposer_id, "0" * 64, len(prompt), 0, False, self._reason, self._detail
            )
        return fl.ProposalCall(
            self.proposer_id, "0" * 64, len(prompt), len(self._sources), True, "", ""
        )

    def programs(self):
        return self._sources if self._left <= 0 else ()


def test_a_persistent_failure_names_the_model_that_failed(tmp_path: Path) -> None:
    ensemble, _ = _two_model((), ())
    delegates = {
        "alpha": FailingProposer((_program("n"),), "alpha", "provider_error", "unauthorized", 9),
        "beta": ScriptedProposer((_program("n * n"),), "beta"),
    }
    problem = fl.declared_problems()["blinded_sequence_rule"]
    config = fl.LoopConfig(generations=4, islands=1, proposals_per_call=2, reset_period=0)
    governor = fl.SpendGovernor(tmp_path / "ledger.json", 500, 5000, 1)
    proposer = me.EnsembleProposer(ensemble, delegates, max_calls=20)
    with pytest.raises(fl.ProposerCallFailed) as caught:
        me.run_ensemble_problem(problem, config, proposer, governor)
    assert "'alpha'" in str(caught.value)
    assert "persistent" in str(caught.value)


def test_a_transient_failure_retries_against_the_next_scheduled_model(tmp_path: Path) -> None:
    """Retrying the model that just failed would put an unscheduled call on it."""

    ensemble, _ = _two_model((), ())
    delegates = {
        "alpha": FailingProposer((_program("n"),), "alpha", "provider_error", "rate limit", 1),
        "beta": ScriptedProposer((_program("n * n"),), "beta"),
    }
    problem = fl.declared_problems()["blinded_sequence_rule"]
    config = fl.LoopConfig(
        generations=2,
        islands=1,
        proposals_per_call=2,
        reset_period=0,
        transient_retries=2,
        retry_backoff_seconds=0.0,
    )
    governor = fl.SpendGovernor(tmp_path / "ledger.json", 500, 5000, 1)
    proposer = me.EnsembleProposer(ensemble, delegates, max_calls=20)
    block = me.run_ensemble_problem(problem, config, proposer, governor)
    drawn = [call["model"] for call in block["proposal_calls"]]
    assert drawn[0] == "alpha" and not block["proposal_calls"][0]["ok"]
    assert drawn[1] == "beta" and block["proposal_calls"][1]["ok"]
    # The failed call was still charged and still counted, so the ledger never lags reality.
    assert sum(call["charged_hundredths"] for call in block["proposal_calls"]) == len(drawn)
    assert fl.read_ledger(tmp_path / "ledger.json") == len(drawn)
    me.validate_block(dict(block, run_label="ensemble"))


def test_the_governor_default_unit_is_unchanged(tmp_path: Path) -> None:
    """The ensemble adds a unit price; it does not change what one plain call costs."""

    governor = fl.SpendGovernor(tmp_path / "ledger.json", 10, 100, 3)
    governor.charge()
    assert governor.charged_hundredths == 3
    governor.charge(4)
    assert governor.charged_hundredths == 3 + 12
    with pytest.raises(fl.FunSearchError):
        governor.charge(0)
    with pytest.raises(fl.FunSearchError):
        governor.may_call(-1)


# ---------------------------------------------------------------------------
# The campaign and its receipt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def campaign(tmp_path_factory) -> dict:
    path = tmp_path_factory.mktemp("ensemble") / "ledger.json"
    result = me.run_ensemble_campaign(
        config=fl.LoopConfig(
            islands=3, generations=12, proposals_per_call=4, reset_period=6, seed=5
        ),
        ledger_path=path,
        max_calls=200,
        max_dollars_hundredths=1000,
    )
    me.validate_receipt(result)
    return result


def test_the_campaign_seals_both_arms(campaign: dict) -> None:
    labels = [block["run_label"] for block in campaign["problems"]]
    assert labels == ["ensemble", "single_model_control"]


def test_the_control_arm_spends_the_same_dollars_not_the_same_calls(campaign: dict) -> None:
    """Equal calls would be a rigged comparison; equal spend is the question worth asking."""

    comparison = campaign["headline"]["ensemble_versus_single_model"]
    assert comparison["spend_is_equal"]
    assert comparison["single_model_calls"] > comparison["ensemble_calls"]


def test_the_ensemble_arm_honours_its_declared_weights(campaign: dict) -> None:
    block = campaign["problems"][0]
    allocation = block["model_yield"]["realized_allocation"]
    assert block["model_yield"]["allocation_honours_declared_weights"]
    assert allocation["broad"] == 3 * allocation["deep"]


def test_the_receipt_carries_the_declared_claims(campaign: dict) -> None:
    assert campaign["claims"] == me.CLAIMS
    assert campaign["claims"]["counterfactual_is_on_the_sealed_population_not_a_rerun"] is True
    assert campaign["claims"]["behaviour_clustering_can_split_never_merge"] is True


def test_the_receipt_holds_no_floats(campaign: dict) -> None:
    """canonical_sha256 forbids floats; this is the check that the receipt never had one."""

    from sigma_theory_compiler.sigma_core import canonical_json_bytes

    assert canonical_json_bytes(campaign)


def _reseal(value: dict) -> dict:
    from sigma_theory_compiler.sigma_core import canonical_sha256

    body = {k: v for k, v in value.items() if k != "content_sha256"}
    core = {
        k: v
        for k, v in value.items()
        if k not in {"content_sha256", "result_core_sha256", "measurement"}
    }
    body["result_core_sha256"] = canonical_sha256(core)
    return {**body, "content_sha256": canonical_sha256(body)}


def test_tampering_with_the_seal_is_detected(campaign: dict) -> None:
    forged = json.loads(json.dumps(campaign))
    forged["problems"][0]["headline"]["best_quality"] = "1.000000000"
    with pytest.raises(me.EnsembleError):
        me.validate_receipt(forged)


def test_a_resealed_forgery_of_the_attribution_is_still_rejected(campaign: dict) -> None:
    """Re-sealing must not launder a lie: the arithmetic is checked, not only the hash."""

    forged = json.loads(json.dumps(campaign))
    for record in forged["problems"][0]["sealed_programs"]:
        if record["attribution"]["proposed_by"]:
            record["attribution"]["proposed_by"] = "a_model_nobody_declared"
            break
    with pytest.raises(me.EnsembleError):
        me.validate_receipt(_reseal(forged))


def test_a_resealed_forgery_of_the_first_seal_counts_is_rejected(campaign: dict) -> None:
    forged = json.loads(json.dumps(campaign))
    forged["problems"][0]["model_yield"]["rows"][0]["programs_sealed_first"] += 1
    with pytest.raises(me.EnsembleError):
        me.validate_receipt(_reseal(forged))


def test_a_resealed_forgery_of_the_allocation_is_rejected(campaign: dict) -> None:
    forged = json.loads(json.dumps(campaign))
    block = forged["problems"][0]
    allocation = block["model_yield"]["realized_allocation"]
    allocation["broad"], allocation["deep"] = allocation["deep"], allocation["broad"]
    with pytest.raises(me.EnsembleError):
        me.validate_receipt(_reseal(forged))


def test_a_model_charged_nothing_for_its_programs_is_rejected(campaign: dict) -> None:
    forged = json.loads(json.dumps(campaign))
    block = forged["problems"][0]
    row = next(item for item in block["model_yield"]["rows"] if item["programs_returned"] > 0)
    row["charged_hundredths"] = 0
    block["charged_hundredths_by_model"][row["model"]] = 0
    with pytest.raises(me.EnsembleError):
        me.validate_receipt(_reseal(forged))


def test_a_forged_ablation_verdict_is_rejected(campaign: dict) -> None:
    forged = json.loads(json.dumps(campaign))
    row = forged["problems"][0]["ablation"]["rows"][0]
    row["cost_verdict"] = (
        "earned_its_cost" if row["cost_verdict"] != "earned_its_cost" else "did_not_earn_its_cost"
    )
    with pytest.raises(me.EnsembleError):
        me.validate_receipt(_reseal(forged))


def test_a_forged_quality_counterfactual_is_rejected(campaign: dict) -> None:
    forged = json.loads(json.dumps(campaign))
    row = forged["problems"][0]["ablation"]["rows"][0]
    row["best_quality_without_this_model"] = "0.000000000"
    with pytest.raises(me.EnsembleError):
        me.validate_receipt(_reseal(forged))


def test_a_receipt_missing_its_control_arm_is_rejected(campaign: dict) -> None:
    forged = json.loads(json.dumps(campaign))
    forged["problems"] = [forged["problems"][0]]
    with pytest.raises(me.EnsembleError):
        me.validate_receipt(_reseal(forged))


def test_the_campaign_is_reproducible(tmp_path: Path) -> None:
    """Same seed, same declaration, same sealed population.  Nothing depends on a clock.

    The comparison is over the ``problems`` blocks rather than the whole receipt because the
    receipt records where the ledger lives, and two runs cannot share one ledger file without
    the second inheriting the first's opening balance.
    """

    from sigma_theory_compiler.sigma_core import canonical_sha256

    config = fl.LoopConfig(
        islands=2, generations=6, proposals_per_call=3, reset_period=0, seed=17
    )
    first = me.run_ensemble_campaign(config=config, ledger_path=tmp_path / "a.json")
    second = me.run_ensemble_campaign(config=config, ledger_path=tmp_path / "b.json")
    assert canonical_sha256(first["problems"]) == canonical_sha256(second["problems"])
    assert first["headline"] == second["headline"]
    assert first["ensemble_sha256"] == second["ensemble_sha256"]


def test_the_mock_campaign_never_reaches_a_provider(tmp_path: Path, monkeypatch) -> None:
    """The whole suite is free, and this is the gate that keeps it that way.

    The sandbox runs every program in a subprocess, so banning ``subprocess`` outright would
    ban the measurement too.  What must never happen is a *provider* call, so the live adapter
    itself is what is booby-trapped.
    """

    def explode(*args, **kwargs):
        raise AssertionError("the test suite must never invoke a provider")

    monkeypatch.setattr(fl.ClaudeCliProposer, "propose", explode)
    monkeypatch.setattr(me, "build_claude_ensemble", explode)
    result = me.run_ensemble_campaign(
        config=fl.LoopConfig(
            islands=2, generations=4, proposals_per_call=2, reset_period=0, seed=3
        ),
        ledger_path=tmp_path / "ledger.json",
    )
    me.validate_receipt(result)


def test_an_undeclared_proposer_kind_is_refused(tmp_path: Path) -> None:
    with pytest.raises(me.EnsembleError):
        me.run_ensemble_campaign(ledger_path=tmp_path / "ledger.json", proposer_kind="telepathy")


# ---------------------------------------------------------------------------
# The sealed receipt on disk, and the CLI
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sealed() -> dict:
    if not RECEIPT.exists():
        pytest.skip("receipt not present")
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_sealed_receipt_validates(sealed: dict) -> None:
    me.validate_receipt(sealed)


def test_sealed_receipt_reports_per_model_yield(sealed: dict) -> None:
    block = next(item for item in sealed["problems"] if item["run_label"] == "ensemble")
    rows = {row["model"]: row for row in block["model_yield"]["rows"]}
    assert set(rows) == {"broad", "deep"}
    for row in rows.values():
        assert row["programs_returned"] > 0
        assert row["distinct_behaviours_contributed"] > 0
        assert "/" in row["declared_share"]
        assert "/" in row["spend_share"]


def test_sealed_receipt_answers_the_cost_question(sealed: dict) -> None:
    """The point of the module: the expensive model's cost is adjudicated, not assumed."""

    block = next(item for item in sealed["problems"] if item["run_label"] == "ensemble")
    rows = {row["model"]: row for row in block["ablation"]["rows"]}
    assert rows["deep"]["cost_verdict"] in {"earned_its_cost", "did_not_earn_its_cost"}
    spend = Fraction(*(int(p) for p in rows["deep"]["spend_share"].split("/")))
    unique = Fraction(*(int(p) for p in rows["deep"]["unique_behaviour_share"].split("/")))
    assert rows["deep"]["unique_share_at_least_spend_share"] == (unique >= spend)


def test_the_registry_entry_is_projected_from_the_receipt(sealed: dict) -> None:
    """The doc quotes numbers; this is what stops them drifting from the computation."""

    document = (ROOT / "docs" / "GOALS_AND_MEASURED_OUTCOMES.md").read_text(encoding="utf-8")
    row = next(line for line in document.splitlines() if line.startswith("| 46 |"))
    comparison = sealed["headline"]["ensemble_versus_single_model"]
    block = next(item for item in sealed["problems"] if item["run_label"] == "ensemble")
    rows = {item["model"]: item for item in block["ablation"]["rows"]}
    assert sealed["content_sha256"][:13] in row
    assert f"{comparison['ensemble_distinct_behaviours']} behaviours" in row
    assert str(comparison["single_model_distinct_behaviours"]) in row
    assert comparison["ensemble_best_quality"] in row
    assert comparison["single_model_best_quality"] in row
    assert f"{block['headline']['charged_hundredths']} simulated" in row
    assert f"{rows['deep']['spend_share']} of the spend for " in row
    assert rows["deep"]["cost_verdict"] in row
    assert rows["broad"]["unique_behaviour_share"] in row
    written = Path(__file__).read_text(encoding="utf-8").count("\ndef test_")
    assert f"{written} tests, zero provider calls, zero real spend" in row


def test_cli_validate_checked() -> None:
    if not RECEIPT.exists():
        pytest.skip("receipt not present")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sigma_theory_compiler.model_ensemble_proposer",
            "--validate-checked",
            "--output",
            str(RECEIPT),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["validated"] is True
