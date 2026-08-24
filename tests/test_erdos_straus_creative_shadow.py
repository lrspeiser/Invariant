from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.claude_creativity_api import CLAUDE_OUTPUT_SCHEMA_VERSION
from sigma_theory_compiler.durable_llm_attempt_journal import DurableAttemptJournal
from sigma_theory_compiler.erdos_straus_creative_shadow import (
    _creative_calls,
    _esdsl2_semantics_contract,
    _file_sha256,
    _mutated_pairs,
    _mutation_lineage,
    _public_payload,
    _run_pairs,
    _run_pairs_fixed_lane_budget,
    _run_pairs_with_attribution,
    _schedule_pairs,
    _witness_sample,
    parse_recipe,
    validate_receipt,
)
from sigma_theory_compiler.exponent_diophantine_sweeper import (
    _es_hard_members,
    es_witness_is_exact,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

EXPERIMENT = {
    "maximum_moduli_per_recipe": 4,
    "maximum_offset": 255,
    "maximum_offsets_per_axis": 6,
    "mutation_radius": 2,
}
ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "runs" / "math" / "erdos-straus-creative-shadow" / "live-runtime.json"
CONFIG_PATH = ROOT / "configs" / "erdos_straus_creative_shadow.json"
MODEL = "claude-opus-4-6"


class SimulatedCrash(RuntimeError):
    pass


class _ExplodingResponse(dict):
    def get(self, key, default=None):
        if key == "type":
            raise SimulatedCrash("process stopped after the durable response append")
        return super().get(key, default)


class ShadowTransport:
    def __init__(self, *, crash_after_first_response: bool = False, invalid_last_critic: bool = False):
        self.requests: list[dict[str, Any]] = []
        self.crash_after_first_response = crash_after_first_response
        self.invalid_last_critic = invalid_last_critic
        self.critic_calls = 0

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, Any]]:
        parsed = None if body is None else json.loads(body)
        self.requests.append(
            {
                "body": parsed,
                "headers": dict(headers),
                "method": method,
                "timeout": timeout,
                "url": url,
            }
        )
        if method == "GET":
            return 200, {
                "capabilities": {"structured_outputs": {"supported": True}},
                "id": MODEL,
                "type": "model",
            }
        prompt = json.loads(parsed["messages"][0]["content"])
        benchmark_id = parsed["output_config"]["format"]["schema"]["properties"][
            "benchmark_id"
        ]["const"]
        role = prompt["role"]
        if role == "critic":
            self.critic_calls += 1
            if self.invalid_last_critic and self.critic_calls == 2:
                output = {}
            else:
                output = {
                    "benchmark_id": benchmark_id,
                    "hypotheses": {},
                    "role": role,
                    "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
                    "steering_actions": {
                        item["candidate_id"]: {
                            "blocker_kind": "bounded_test_blocker",
                            "distance_denominator": 2,
                            "distance_numerator": 1,
                            "repair": "Retain and recombine with a different offset scale.",
                            "verdict": "repair",
                        }
                        for item in prompt["candidate_summaries"]
                    },
                }
        else:
            requested = 2 if self.invalid_last_critic else 1
            output = {
                "benchmark_id": benchmark_id,
                "hypotheses": [self._hypothesis(index) for index in range(requested)],
                "role": role,
                "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
                "steering_actions": {},
            }
        response: Mapping[str, Any] = {
            "content": [{"text": json.dumps(output), "type": "text"}],
            "id": f"msg.shadow.{len(self.requests)}",
            "model": MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        if self.crash_after_first_response:
            self.crash_after_first_response = False
            response = _ExplodingResponse(response)
        return 200, response

    @staticmethod
    def _hypothesis(index: int) -> dict[str, Any]:
        return {
            "expression": f"ESDSL1|basis=lattice_transform|x=0,{index + 1}|t=0,{index + 2}|m=24",
            "falsifiers": ["Fails an exact modular lane test."],
            "family": f"shadow_family_{index}",
            "hypothesis_id": f"shadow.{index}",
            "invariants": ["Exact integer divisibility only."],
            "known_analogues": ["lattice parameterization"],
            "llm_origin_assessment": "cross_domain_synthesis",
            "proof_plan": ["Derive the divisor identity independently."],
            "rationale": "A bounded fake response for durable recovery testing.",
            "representation": "modular_relation",
            "source_idea_domains": ["lattices", "Egyptian fractions"],
            "synthesis_note": "Novelty remains unestablished pending prior-art review.",
        }


class NoCallTransport:
    def __init__(self):
        self.calls = 0

    def __call__(self, *_args):
        self.calls += 1
        raise AssertionError("a completed scheduled slot was redispatched")


def _test_config(*, critics: bool) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = deepcopy(config)
    config["campaign_id"] = f"erdos-straus-journal-test-{int(critics)}"
    config["experiment"]["creative_roles"] = ["proposer"]
    config["experiment"]["requested_ideas_per_creative_call"] = 2 if critics else 1
    config["experiment"]["llm_critic_batch_size"] = 1 if critics else 0
    return config


def _journal(tmp_path: Path, config: Mapping[str, Any]) -> DurableAttemptJournal:
    return DurableAttemptJournal.create(
        tmp_path / "private" / "attempts.jsonl",
        experiment_id=config["campaign_id"],
        source_bindings={"test_source_sha256": "a" * 64},
        unblinding_key=b"j" * 32,
    )


def _with_test_credential(callback):
    original = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "secret-never-persisted"
    try:
        return callback()
    finally:
        if original is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = original


def test_recipe_parser_accepts_typed_schedule_and_rejects_prose():
    recipe = parse_recipe("ESDSL1|basis=lattice_transform|x=0,2,65|t=0,7|m=24,120", EXPERIMENT)
    assert recipe == {
        "basis": "lattice_transform",
        "moduli": [24, 120],
        "t_offsets": [0, 7],
        "x_offsets": [0, 2, 65],
    }
    assert parse_recipe("try a clever lattice", EXPERIMENT) is None
    assert parse_recipe("ESDSL1|basis=magic|x=0|t=0|m=24", EXPERIMENT) is None
    assert parse_recipe("ESDSL1|basis=divisor_pair|x=999|t=0|m=24", EXPERIMENT) is None


def test_esdsl2_basis_operators_have_distinct_exact_schedules_and_matched_controls():
    contract = _esdsl2_semantics_contract(EXPERIMENT)
    assert contract["schema_version"] == "invariant-esdsl2-basis-semantics-contract-1.0"
    assert contract["all_basis_schedules_nonempty"] is True
    assert contract["basis_schedule_sha256s_unique"] is True
    assert [item["basis"] for item in contract["basis_controls"]] == [
        "continued_fraction",
        "descent_graph",
        "divisor_pair",
        "greedy_offset",
        "lattice_transform",
        "modular_sieve",
        "polynomial_ansatz",
        "residue_cover",
    ]
    assert {
        item["basis"]: item["schedule_pairs"] for item in contract["basis_controls"]
    } == {
        "continued_fraction": [[1, 1], [3, 2], [7, 5], [17, 12]],
        "descent_graph": [[4, 4], [3, 6], [6, 3], [2, 8], [5, 5], [8, 2]],
        "divisor_pair": [[2, 14], [3, 8], [4, 6], [2, 20], [3, 11], [4, 8]],
        "greedy_offset": [[0, 1], [4, 1], [0, 5], [4, 5], [9, 1]],
        "lattice_transform": [[1, 2], [3, 8], [3, 3], [5, 9]],
        "modular_sieve": [
            [0, 0],
            [0, 2],
            [1, 1],
            [1, 3],
            [2, 0],
            [2, 2],
            [3, 1],
            [3, 3],
        ],
        "polynomial_ansatz": [[1, 0], [2, 1], [3, 4], [4, 9]],
        "residue_cover": [[1, 4], [6, 9], [2, 3], [7, 8], [4, 1], [9, 6]],
    }
    for item in contract["basis_controls"]:
        assert item["pair_count_matched"] is True
        assert item["grammar_field_count_matched"] is True
        assert item["verifier_lane_budget_matched"] is True
        assert item["schedule_pairs"] != item["matched_control_pairs"]


def test_esdsl2_schedules_feed_the_exact_witness_checker():
    members = _es_hard_members(10_000)
    for item in _esdsl2_semantics_contract(EXPERIMENT)["basis_controls"]:
        pairs = [tuple(pair) for pair in item["schedule_pairs"]]
        wx, wy, resolved, lane_tests, _ = _run_pairs(__import__("numpy"), members, pairs)
        assert lane_tests > 0
        assert int(resolved.sum()) > 0
        assert _witness_sample(members, wx, wy, resolved)


def test_esdsl2_campaign_prompt_exposes_every_strict_basis_form():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    legacy_machine_recipe = _public_payload(config)["machine_recipe"]
    assert "exactly ESDSL1|basis=B|x=X|t=T|m=M" in legacy_machine_recipe
    assert "ESDSL2" not in legacy_machine_recipe
    config["experiment"]["proposal_dsl_version"] = "ESDSL2"
    machine_recipe = _public_payload(config)["machine_recipe"]
    assert "ESDSL2|basis=B" in machine_recipe
    for basis in (
        "continued_fraction",
        "descent_graph",
        "divisor_pair",
        "greedy_offset",
        "lattice_transform",
        "modular_sieve",
        "polynomial_ansatz",
        "residue_cover",
    ):
        assert basis in machine_recipe


@pytest.mark.parametrize(
    "expression",
    [
        "ESDSL2|basis=continued_fraction|scale=1|a=1,2|m=24",
        "ESDSL2|basis=descent_graph|start=4,4|moves=0,0|depth=2|m=24",
        "ESDSL2|basis=divisor_pair|n=12|shift=-1,0|m=24",
        "ESDSL2|basis=greedy_offset|x=0,1|t=0,1|budget=99|m=24",
        "ESDSL2|basis=lattice_transform|u=0,1|v=0,1|matrix=1,2,2,4|shift=0,0|m=24",
        "ESDSL2|basis=modular_sieve|x=0|t=0|congruence=1,1,1,2|m=24",
        "ESDSL2|basis=polynomial_ansatz|k=16|xcoef=16,16,16|tcoef=0|m=24",
        "ESDSL2|basis=residue_cover|q=5|residues=5|lifts=0|m=24",
        "ESDSL2|basis=residue_cover|q=5|residues=1|lifts=0|m=24|m=30",
        "ESDSL2|basis=greedy_offset|x=1,0|t=0,1|budget=2|m=24",
        "ESDSL2|basis=greedy_offset|x=0,0|t=0,1|budget=2|m=24",
        "ESDSL2|basis=continued_fraction|a=1,2|scale=01|m=24",
        "ESDSL2|basis=descent_graph|start=4,4|moves=-0,1|depth=2|m=24",
    ],
)
def test_esdsl2_rejects_noncanonical_degenerate_or_out_of_bounds_recipes(expression: str):
    assert parse_recipe(expression, EXPERIMENT) is None


def test_esdsl1_parse_shape_and_cartesian_order_are_immutable():
    expression = "ESDSL1|basis=lattice_transform|x=0,2,65|t=0,7|m=24,120"
    recipe = parse_recipe(expression, EXPERIMENT)
    assert recipe == {
        "basis": "lattice_transform",
        "moduli": [24, 120],
        "t_offsets": [0, 7],
        "x_offsets": [0, 2, 65],
    }
    assert _schedule_pairs(recipe or {}) == ((0, 0), (0, 7), (2, 0), (2, 7), (65, 0), (65, 7))


def test_source_binding_is_portable_across_lf_and_crlf(tmp_path: Path):
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"first\nsecond\n")
    crlf.write_bytes(b"first\r\nsecond\r\n")
    assert _file_sha256(lf) == _file_sha256(crlf)


def test_exact_pair_schedule_produces_replayable_witnesses():
    members = _es_hard_members(10_000)
    wx, wy, resolved, lane_tests, _ = _run_pairs(__import__("numpy"), members, [(0, 0), (1, 0)])
    assert lane_tests > 0
    assert int(resolved.sum()) > 0
    assert _witness_sample(members, wx, wy, resolved)


def test_exact_pair_schedule_records_the_first_successful_pair():
    members = _es_hard_members(10_000)
    pairs = [(0, 0), (1, 0), (2, 1)]
    wx, wy, resolved, winning_dx, winning_t, _, _ = _run_pairs_with_attribution(
        __import__("numpy"), members, pairs
    )
    for index in __import__("numpy").flatnonzero(resolved):
        winning_pair = (int(winning_dx[index]), int(winning_t[index]))
        assert winning_pair in pairs
        n, x, y = int(members[index]), int(wx[index]), int(wy[index])
        b = n * x
        d = (4 * x - n) * y - b
        assert d > 0 and (b * y) % d == 0
        assert es_witness_is_exact(n, x, y, (b * y) // d)


def test_fixed_lane_budget_matches_first_success_results_and_executes_every_lane():
    np = __import__("numpy")
    members = _es_hard_members(10_000)
    pairs = [(0, 0), (1, 0), (2, 1), (64, 32)]
    early_wx, early_wy, early_resolved, early_lane_tests, _ = _run_pairs(np, members, pairs)
    fixed_wx, fixed_wy, fixed_resolved, fixed_lane_tests, _ = _run_pairs_fixed_lane_budget(
        np, members, pairs
    )
    assert np.array_equal(fixed_resolved, early_resolved)
    assert np.array_equal(fixed_wx, early_wx)
    assert np.array_equal(fixed_wy, early_wy)
    assert fixed_lane_tests == len(members) * len(pairs)
    assert early_lane_tests <= fixed_lane_tests
    assert _witness_sample(members, fixed_wx, fixed_wy, fixed_resolved)


def test_mutation_preserves_direct_pairs_and_adds_neighbors():
    pairs = _mutated_pairs([[64, 32]], EXPERIMENT)
    assert (64, 32) in pairs
    assert (62, 30) in pairs
    assert (66, 34) in pairs
    assert len(pairs) == 25


def test_mutation_lineage_preserves_overlapping_parent_ideas():
    creative = {
        "ideas": [
            {
                "execution": {
                    "admission": "EXECUTED_EXACT_MODULAR_SCREEN",
                    "direct_parameter_pairs": [[1, 2]],
                    "recipe": {"basis": "lattice_transform"},
                },
                "idea_id": "idea.one",
                "llm_self_assessed_origin": "cross_domain_synthesis",
                "role": "recombiner",
            },
            {
                "execution": {
                    "admission": "EXECUTED_EXACT_MODULAR_SCREEN",
                    "direct_parameter_pairs": [[2, 2]],
                    "recipe": {"basis": "divisor_pair"},
                },
                "idea_id": "idea.two",
                "llm_self_assessed_origin": "known_rewrite",
                "role": "recombiner",
            },
        ]
    }
    lineage = _mutation_lineage(creative, EXPERIMENT)
    parents = lineage[(2, 2)]
    assert {(item["idea_id"], tuple(item["mutation_delta"])) for item in parents} == {
        ("idea.one", (1, 0)),
        ("idea.two", (0, 0)),
    }
    assert {item["basis"] for item in parents} == {"lattice_transform", "divisor_pair"}


def test_response_survives_process_stop_and_replays_without_redispatch(tmp_path: Path):
    config = _test_config(critics=False)
    journal = _journal(tmp_path, config)
    first = ShadowTransport(crash_after_first_response=True)
    with pytest.raises(SimulatedCrash, match="durable response"):
        _with_test_credential(
            lambda: _creative_calls(config, journal, base_transport=first)
        )
    loaded = DurableAttemptJournal.load(journal.path)
    assert loaded.event_counts() == {
        "journal_header": 1,
        "message_dispatch": 1,
        "message_response": 1,
        "model_probe_dispatch": 1,
        "model_probe_response": 1,
    }
    resumed_transport = NoCallTransport()
    ideas, evidence = _with_test_credential(
        lambda: _creative_calls(config, loaded, base_transport=resumed_transport)
    )
    assert resumed_transport.calls == 0
    assert len(ideas) == 1
    assert evidence["budget"] == {
        "calls": 1,
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    assert evidence["attempts"][0]["status"] == "completed"
    assert "secret-never-persisted" not in journal.path.read_text(encoding="utf-8")


def test_missing_credential_does_not_permanently_consume_a_slot(tmp_path: Path):
    config = _test_config(critics=False)
    journal = _journal(tmp_path, config)
    original = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="no journaled Claude idea"):
            _creative_calls(config, journal, base_transport=NoCallTransport())
    finally:
        if original is not None:
            os.environ["ANTHROPIC_API_KEY"] = original
    assert journal.event_counts() == {"journal_header": 1}
    transport = ShadowTransport()
    ideas, evidence = _with_test_credential(
        lambda: _creative_calls(config, journal, base_transport=transport)
    )
    assert len(ideas) == 1
    assert evidence["budget"]["calls"] == 1


def test_restart_accepts_repeated_identical_model_probes_and_recovers_critic(tmp_path: Path):
    config = _test_config(critics=False)
    journal = _journal(tmp_path, config)
    _with_test_credential(
        lambda: _creative_calls(config, journal, base_transport=ShadowTransport())
    )
    config["experiment"]["llm_critic_batch_size"] = 1
    with pytest.raises(SimulatedCrash, match="durable response"):
        _with_test_credential(
            lambda: _creative_calls(
                config,
                DurableAttemptJournal.load(journal.path),
                base_transport=ShadowTransport(crash_after_first_response=True),
            )
        )
    loaded = DurableAttemptJournal.load(journal.path)
    assert loaded.event_counts()["model_probe_response"] == 2
    no_calls = NoCallTransport()
    ideas, evidence = _with_test_credential(
        lambda: _creative_calls(config, loaded, base_transport=no_calls)
    )
    assert no_calls.calls == 0
    assert evidence["budget"]["calls"] == 2
    assert ideas[0]["critic_source"] == "journaled_llm_critic_retained_without_pruning"


def test_later_bad_critic_is_retained_and_all_slots_resume_without_calls(tmp_path: Path):
    config = _test_config(critics=True)
    journal = _journal(tmp_path, config)
    first = ShadowTransport(invalid_last_critic=True)
    ideas, evidence = _with_test_credential(
        lambda: _creative_calls(config, journal, base_transport=first)
    )
    assert len([item for item in first.requests if item["method"] == "POST"]) == 3
    assert evidence["budget"]["calls"] == 3
    assert [item["status"] for item in evidence["attempts"]] == [
        "completed",
        "completed",
        "client_or_contract_failure",
    ]
    assert ideas[0]["critic_source"] == "journaled_llm_critic_retained_without_pruning"
    assert ideas[1]["critic_source"] == "deterministic_machine_admission_not_llm_critique"
    assert all(item["critic"]["verdict"] != "reject" for item in ideas)
    resumed_transport = NoCallTransport()
    resumed_ideas, resumed_evidence = _with_test_credential(
        lambda: _creative_calls(
            config,
            DurableAttemptJournal.load(journal.path),
            base_transport=resumed_transport,
        )
    )
    assert resumed_transport.calls == 0
    assert resumed_ideas == ideas
    assert resumed_evidence == evidence


def test_shipped_live_receipt_validates_and_preserves_claim_boundary():
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    validate_receipt(receipt, ROOT)
    assert receipt["status"] == "PASS_BOUNDED_CREATIVE_SHADOW_NO_OPEN_PROBLEM_CLAIM"
    typed_compiler = receipt["typed_schedule_compiler"]
    assert typed_compiler == _esdsl2_semantics_contract(EXPERIMENT)
    assert len(typed_compiler["basis_controls"]) == 8
    assert all(value is False for value in typed_compiler["claim_boundary"].values())
    assert receipt["accounting"] == {
        "baseline_gpu_lane_tests": 104_839_060,
        "creative_tail_lane_tests": 344_279,
        "denominators_covered": 99_999_999,
        "executable_llm_ideas": 11,
        "llm_ideas_proposed": 12,
        "llm_provider_calls": 4,
        "matched_control_lane_tests": 36_205_832,
        "mutated_parameter_pairs": 1_051,
        "retained_llm_provider_calls": 1,
        "total_exact_modular_lane_tests": 148_875_850,
    }
    assert receipt["hard_tail_funnel"]["creative_tail"]["resolved_from_baseline_tail"] == 173
    assert receipt["hard_tail_funnel"]["creative_tail"]["independent_cpu_exact_verified"] == 173
    controls = receipt["hard_tail_funnel"]["matched_random_controls"]
    assert controls["fixed_lane_evaluator"] is True
    assert controls["early_stop_enabled"] is False
    assert controls["exact_lane_budget_matched"] is True
    assert controls["wall_clock_budget_claimed_matched"] is True
    assert controls["same_device_and_evaluator_kernel"] is True
    assert controls["wall_clock_ceiling_seconds"] == 30
    assert controls["random_control_exact_lane_tests"] == 35_832_576
    assert controls["total_exact_lane_tests"] == 36_205_832
    fixed_reference = controls["creative_fixed_lane_reference"]
    assert fixed_reference["exact_lane_tests"] == 373_256
    assert fixed_reference["resolved"] == 173
    assert fixed_reference["first_success_result_agreement"] is True
    assert float(fixed_reference["elapsed_seconds"]) <= 30
    for key in ("uniform_domain", "support_matched", "pairing_only_rewire"):
        assert controls[key]["exact_lane_tests_per_trial"] == 373_256
        assert controls[key]["total_exact_lane_tests"] == 11_944_192
        assert controls[key]["all_trials_within_wall_clock_ceiling"] is True
        assert len(controls[key]["elapsed_seconds"]) == 32
        assert max(float(item) for item in controls[key]["elapsed_seconds"]) <= 30
    rewired = controls["pairing_only_rewire"]
    assert rewired["median_resolved"] == "174.000000"
    assert rewired["random_at_least_creative"] == 24
    attribution = receipt["hard_tail_funnel"]["creative_tail"][
        "parent_lineage_attribution"
    ]
    assert attribution["all_resolved_hits_have_parent_lineage"] is True
    assert len(attribution["resolved_hit_records"]) == 173
    assert attribution["multi_parent_lineage_hits"] == 27
    assert attribution["multi_idea_lineage_hits"] == 0
    assert attribution["multi_basis_lineage_hits"] == 0
    assert {
        item["idea_id"]: item["linked_resolved_hits"]
        for item in attribution["idea_linked_hits"]
    } == {
        "recombiner.02.ES-CF-002": 33,
        "recombiner.08.ES-MS-008": 51,
        "recombiner.09.ES-DC-009": 18,
        "recombiner.10.ES-HL-010": 1,
        "recombiner.12.ES-TP-012": 70,
    }
    example = next(
        item for item in attribution["resolved_hit_records"] if item["n"] == 398_161
    )
    assert example["winning_pair"] == [83, 11]
    assert example["parent_lineages"] == [
        {
            "basis": "polynomial_ansatz",
            "direct_pair": [81, 13],
            "idea_id": "recombiner.12.ES-TP-012",
            "llm_self_assessed_origin": "proposed_new_construction",
            "mutation_delta": [2, -2],
            "role": "recombiner",
        }
    ]
    assert all(value is False for value in receipt["claim_boundary"].values())

    tampered = deepcopy(receipt)
    tampered_record = tampered["hard_tail_funnel"]["creative_tail"][
        "parent_lineage_attribution"
    ]["resolved_hit_records"][0]
    tampered_record["winning_pair"][0] += 1
    tampered["content_sha256"] = canonical_sha256(
        {key: item for key, item in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="lineage"):
        validate_receipt(tampered, ROOT)

    tampered = deepcopy(receipt)
    tampered["typed_schedule_compiler"]["basis_controls"][0]["schedule_pairs"][0][0] += 1
    tampered["content_sha256"] = canonical_sha256(
        {key: item for key, item in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="typed schedule compiler"):
        validate_receipt(tampered, ROOT)

    tampered = deepcopy(receipt)
    tampered["hard_tail_funnel"]["matched_random_controls"]["uniform_domain"][
        "exact_lane_tests_per_trial"
    ] -= 1
    tampered["content_sha256"] = canonical_sha256(
        {key: item for key, item in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="fixed-lane random control"):
        validate_receipt(tampered, ROOT)

    tampered = deepcopy(receipt)
    tampered["hard_tail_funnel"]["matched_random_controls"]["support_matched"][
        "elapsed_seconds"
    ][0] = "30.000001"
    tampered["content_sha256"] = canonical_sha256(
        {key: item for key, item in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="fixed-lane random control"):
        validate_receipt(tampered, ROOT)
