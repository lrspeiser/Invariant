from __future__ import annotations

import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler import external_creativity_validation as E
from sigma_theory_compiler.claude_creativity_api import CLAUDE_OUTPUT_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-4-6"


class CampaignClaudeTransport:
    def __init__(
        self,
        proposer_expression: str = "x0",
        proposer_representation: str = "sympy_expression",
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self.proposer_expression = proposer_expression
        self.proposer_representation = proposer_representation

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
            {"body": parsed, "headers": dict(headers), "method": method, "url": url}
        )
        if method == "GET":
            return 200, {
                "capabilities": {"structured_outputs": {"supported": True}},
                "id": MODEL,
                "type": "model",
            }
        assert parsed is not None
        prompt = json.loads(parsed["messages"][0]["content"])
        benchmark_id = prompt["benchmark"]["blind_id"]
        role = prompt["role"]
        if role == "proposer":
            hypothesis = {
                "expression": self.proposer_expression,
                "falsifiers": ["sealed holdout"],
                "family": "analogy_transfer",
                "hypothesis_id": f"hypothesis.{len(self.requests)}",
                "invariants": ["identity_scaling"],
                "known_analogues": ["identity map"],
                "llm_origin_assessment": "known_rewrite",
                "proof_plan": ["test base cases", "induct"],
                "rationale": "A deliberately simple typed control hypothesis.",
                "representation": self.proposer_representation,
                "source_idea_domains": ["algebra", "recurrences"],
                "synthesis_note": "A control recovered through recurrence language.",
            }
            hypothesis_schema = parsed["output_config"]["format"]["schema"]["properties"][
                "hypotheses"
            ]
            if hypothesis_schema["type"] == "object":
                hypotheses = {
                    name: hypothesis | {"hypothesis_id": f"{hypothesis['hypothesis_id']}.{name}"}
                    for name in hypothesis_schema["required"]
                }
            else:
                hypotheses = [hypothesis]
            output = {
                "benchmark_id": benchmark_id,
                "hypotheses": hypotheses,
                "role": role,
                "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
                "steering_actions": [],
            }
        else:
            candidate_id = prompt["candidate_summaries"][0]["candidate_id"]
            output = {
                "benchmark_id": benchmark_id,
                "hypotheses": [],
                "role": role,
                "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
                "steering_actions": [
                    {
                        "blocker_kind": "train_residual",
                        "candidate_id": candidate_id,
                        "distance_denominator": 1,
                        "distance_numerator": 1,
                        "repair": "Change representation and test another invariant.",
                        "verdict": "repair",
                    }
                ],
            }
        return 200, {
            "content": [{"text": json.dumps(output), "type": "text"}],
            "id": f"msg_campaign_{len(self.requests)}",
            "model": MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 200, "output_tokens": 100},
        }


@pytest.fixture(scope="module")
def dry_receipt() -> dict[str, Any]:
    return E.run_campaign(ROOT)


def _hypothesis(
    representation: str,
    expression: str,
    *,
    origin: str = "proposed_new_construction",
) -> E.ClaudeHypothesis:
    return E.ClaudeHypothesis(
        hypothesis_id=f"typed.{representation}",
        family="typed_test",
        representation=representation,
        expression=expression,
        invariants=("typed_test_invariant",),
        proof_plan=("typed_test_plan",),
        falsifiers=("independent evaluator disagreement",),
        rationale="A bounded typed test hypothesis.",
        llm_origin_assessment=origin,
        known_analogues=("test fixture",),
        source_idea_domains=("algebra", "sequences"),
        synthesis_note="Test-only typed synthesis.",
    )


def test_external_authorship_is_distinct_and_generation_view_is_anonymous() -> None:
    public, benchmarks = E.load_public_benchmarks(ROOT)
    assert public["generator_principal_id"] == "invariant.discovery-engine"
    assert len(benchmarks) == 4
    assert [item.capability_level for item in benchmarks].count(4) == 2
    assert [item.capability_level for item in benchmarks].count(5) == 2
    for benchmark in benchmarks:
        assert benchmark.source.authoring_principal_id.startswith("external.")
        assert benchmark.source.authoring_principal_id != public["generator_principal_id"]
        generation = benchmark.generation_view()
        serialized = json.dumps(generation, sort_keys=True)
        assert benchmark.benchmark_id not in serialized
        assert benchmark.source.source_uri not in serialized
        assert "target" not in serialized
        assert "holdout" not in serialized
        assert list(generation["variables"]) == [
            f"x{index}" for index in range(len(benchmark.variables))
        ]


def test_targets_open_commitments_and_bounded_unknown_has_no_formula() -> None:
    public, benchmarks = E.load_public_benchmarks(ROOT)
    targets = E.unseal_targets(ROOT, public, benchmarks)
    assert len(targets) == 4
    for target in targets:
        benchmark = next(item for item in benchmarks if item.benchmark_id == target.benchmark_id)
        assert target.commitment == benchmark.target_commitment
        if target.target_kind == "bounded_unknown":
            assert target.reference_formula is None
        else:
            assert target.reference_formula is not None


def test_target_commitment_tamper_fails_closed(tmp_path: Path) -> None:
    public = json.loads((ROOT / E.PUBLIC_CONFIG_PATH).read_text(encoding="utf-8"))
    targets = json.loads(
        (ROOT / "configs/external_sealed_creativity_targets.json").read_text(encoding="utf-8")
    )
    targets["targets"][0]["holdout_records"][0]["output"] = "101"
    (tmp_path / "configs").mkdir()
    (tmp_path / E.PUBLIC_CONFIG_PATH).write_text(json.dumps(public), encoding="utf-8")
    (tmp_path / public["sealed_targets_path"]).write_text(json.dumps(targets), encoding="utf-8")
    _, benchmarks = E.load_public_benchmarks(tmp_path)
    with pytest.raises(E.ExternalCreativityError, match="does not open"):
        E.unseal_targets(tmp_path, public, benchmarks)


def test_source_bindings_normalize_git_checkout_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"first\nsecond\n")
    crlf.write_bytes(b"first\r\nsecond\r\n")
    assert E._file_sha256(lf) == E._file_sha256(crlf)


def test_claude_arithmetic_normalizer_accepts_bounded_notation_only() -> None:
    normalized, method = E._normalize_claude_arithmetic(
        "output = n*(n+1)*(2*n+1)/6 = (2*n**3 + 3*n**2 + n)/6",
        ("x0",),
    )
    assert E._safe_expression(normalized, ("x0",)) == E._safe_expression(
        "x0*(x0+1)*(2*x0+1)/6", ("x0",)
    )
    assert "output_assignment" in method
    assert "single_variable_alias" in method
    assert "equivalent_equality" in method
    with pytest.raises(E.ExternalCreativityError):
        E._normalize_claude_arithmetic("x0 = x0 + 1", ("x0",))
    with pytest.raises(E.ExternalCreativityError):
        E._normalize_claude_arithmetic("try a polynomial near x0", ("x0",))


def test_typed_claude_compiler_executes_with_independent_agreement() -> None:
    _, benchmarks = E.load_public_benchmarks(ROOT)
    benchmark = next(item for item in benchmarks if len(item.aliases) == 1)
    rows = tuple(E.Observation((Fraction(index),), Fraction(0)) for index in range(7))
    cases = [
        (
            "linear_recurrence",
            '{"coefficients":["1","1"],"seed":["0","1"]}',
            (0, 1, 1, 2, 3, 5, 8),
        ),
        (
            "generating_function",
            ('{"denominator":["1","-1","-1"],"index":"x0","numerator":["0","1"]}'),
            (0, 1, 1, 2, 3, 5, 8),
        ),
        (
            "finite_sum",
            '{"body":"k**2","index":"k","lower":"1","upper":"x0"}',
            (0, 1, 5, 14, 30, 55, 91),
        ),
        (
            "finite_product",
            '{"body":"k","index":"k","lower":"1","upper":"x0"}',
            (1, 1, 2, 6, 24, 120, 720),
        ),
        (
            "modular_relation",
            '{"expression":"x0**2","modulus":5}',
            (0, 1, 4, 4, 1, 0, 1),
        ),
        (
            "piecewise_relation",
            json.dumps(
                {
                    "branches": [
                        {
                            "condition": {"comparator": "lt", "left": "x0", "right": "3"},
                            "expression": "x0 + 10",
                        }
                    ],
                    "default_expression": "x0",
                }
            ),
            (10, 11, 12, 3, 4, 5, 6),
        ),
        (
            "transform_relation",
            json.dumps(
                {
                    "claimed_transform": "3*x0**2 + 3*x0 + 1",
                    "index": "x0",
                    "source_expression": "x0**3",
                    "stencil": [
                        {"coefficient": "-1", "offset": 0},
                        {"coefficient": "1", "offset": 1},
                    ],
                    "transform_kind": "linear_shift_stencil",
                }
            ),
            (1, 7, 19, 37, 61, 91, 127),
        ),
        (
            "tensor_identity",
            json.dumps(
                {
                    "left_components": ["x0", "0", "0", "x0"],
                    "output_component": {"flat_index": 0, "side": "left"},
                    "right_components": ["x0", "0", "0", "x0"],
                    "shape": [2, 2],
                    "symmetries": [{"left_axis": 0, "right_axis": 1, "sign": 1}],
                    "tensor_name": "T",
                    "variance": ["covariant", "covariant"],
                }
            ),
            (0, 1, 2, 3, 4, 5, 6),
        ),
        (
            "variational_principle",
            json.dumps(
                {
                    "bindings": {
                        "q": "0",
                        "q_ddot": "-x0",
                        "q_dot": "0",
                        "t": "0",
                    },
                    "claimed_euler_lagrange": "-q_ddot",
                    "coordinate": "t",
                    "field": "q",
                    "first_derivative": "q_dot",
                    "integrand": "q_dot**2/2",
                    "second_derivative": "q_ddot",
                }
            ),
            (0, 1, 2, 3, 4, 5, 6),
        ),
    ]
    for representation, expression, expected in cases:
        candidate, record = E._claude_candidate(benchmark, _hypothesis(representation, expression))
        assert candidate is not None
        assert record["status"] == "ADMITTED_EXECUTABLE"
        primary = E.predict(candidate, benchmark, rows)
        independent = E.independently_predict(candidate, benchmark, rows)
        assert primary == independent == tuple(Fraction(item) for item in expected)


def test_live_known_rewrite_transform_replays_after_safe_alias_normalization() -> None:
    _, benchmarks = E.load_public_benchmarks(ROOT)
    benchmark = next(
        item for item in benchmarks if item.benchmark_id == "external.authority-nist-0244"
    )
    expression = json.dumps(
        {
            "claimed_transform": "(n+1)^2",
            "index": "n",
            "source_expression": "n*(n+1)*(2*n+1)/6",
            "stencil": [
                {"coefficient": "1", "offset": 1},
                {"coefficient": "-1", "offset": 0},
            ],
            "transform_kind": "linear_shift_stencil",
        }
    )
    candidate, record = E._claude_candidate(
        benchmark,
        _hypothesis("transform_relation", expression, origin="known_rewrite"),
    )
    assert candidate is not None
    assert record["status"] == "ADMITTED_EXECUTABLE"
    assert json.loads(candidate.expression)["index"] == "x0"
    rows = tuple(E.Observation((Fraction(index),), Fraction(0)) for index in range(7))
    expected = tuple(Fraction((index + 1) ** 2) for index in range(7))
    assert E.predict(candidate, benchmark, rows) == expected
    assert E.independently_predict(candidate, benchmark, rows) == expected


def test_piecewise_relation_has_ordered_exact_boundary_semantics() -> None:
    _, benchmarks = E.load_public_benchmarks(ROOT)
    benchmark = next(item for item in benchmarks if len(item.aliases) == 1)
    expression = json.dumps(
        {
            "branches": [
                {
                    "condition": {"comparator": "lt", "left": "x0", "right": "0"},
                    "expression": "-x0",
                },
                {
                    "condition": {"comparator": "eq", "left": "x0", "right": "0"},
                    "expression": "0",
                },
            ],
            "default_expression": "x0",
        }
    )
    candidate, record = E._claude_candidate(
        benchmark, _hypothesis("piecewise_relation", expression)
    )
    assert candidate is not None
    assert record["normalization"] == (
        "canonical_typed_json+ordered_exact_predicates+exact_extended_arithmetic"
    )
    rows = tuple(
        E.Observation((value,), Fraction(0))
        for value in (Fraction(-3, 2), Fraction(-1, 2), Fraction(0), Fraction(1, 2))
    )
    expected = tuple(Fraction(item) for item in (Fraction(3, 2), Fraction(1, 2), 0, Fraction(1, 2)))
    assert E.predict(candidate, benchmark, rows) == expected
    assert E.independently_predict(candidate, benchmark, rows) == expected


def test_live_style_modulo_piecewise_replays_with_independent_agreement() -> None:
    _, benchmarks = E.load_public_benchmarks(ROOT)
    benchmark = next(item for item in benchmarks if len(item.aliases) == 1)
    expression = json.dumps(
        {
            "branches": [
                {
                    "condition": {"comparator": "lt", "left": "x0", "right": "8"},
                    "expression": "x0",
                },
                {
                    "condition": {"comparator": "eq", "left": "x0 % 2", "right": "0"},
                    "expression": "16 - x0 / 2",
                },
                {
                    "condition": {"comparator": "eq", "left": "x0 % 2", "right": "1"},
                    "expression": "(x0 + 33) / 2",
                },
            ],
            "default_expression": "0",
        }
    )
    candidate, record = E._claude_candidate(
        benchmark,
        _hypothesis("piecewise_relation", expression, origin="cross_domain_synthesis"),
    )
    assert candidate is not None
    assert record["status"] == "ADMITTED_EXECUTABLE"
    assert record["llm_self_assessed_origin"] == "cross_domain_synthesis"
    rows = tuple(E.Observation((Fraction(value),), Fraction(0)) for value in (0, 7, 8, 9, 10))
    expected = tuple(Fraction(value) for value in (0, 7, 12, 21, 11))
    assert E.predict(candidate, benchmark, rows) == expected
    assert E.independently_predict(candidate, benchmark, rows) == expected


def test_live_style_floor_conditional_and_decimals_are_exactly_executable() -> None:
    _, benchmarks = E.load_public_benchmarks(ROOT)
    benchmark = next(item for item in benchmarks if len(item.aliases) == 1)
    floor_expression = json.dumps(
        {
            "branches": [
                {
                    "condition": {"comparator": "eq", "left": "x0 % 2", "right": "0"},
                    "expression": ("(x0 // 2) + (x0 // 2) * ((x0 // 2) + 1) // 2 if x0 > 0 else 0"),
                },
                {
                    "condition": {"comparator": "eq", "left": "x0 % 2", "right": "1"},
                    "expression": ("(x0 + 1) // 2 + ((x0 - 1) // 2) * (((x0 - 1) // 2) + 1) // 2"),
                },
            ],
            "default_expression": "0",
        }
    )
    floor_candidate, _ = E._claude_candidate(
        benchmark, _hypothesis("piecewise_relation", floor_expression, origin="uncertain")
    )
    assert floor_candidate is not None
    rows = tuple(E.Observation((Fraction(value),), Fraction(0)) for value in range(6))
    expected = tuple(Fraction(value) for value in (0, 1, 2, 3, 5, 6))
    assert E.predict(floor_candidate, benchmark, rows) == expected
    assert E.independently_predict(floor_candidate, benchmark, rows) == expected

    rounded_expression = json.dumps(
        {
            "branches": [
                {
                    "condition": {"comparator": "le", "left": "x0", "right": "3"},
                    "expression": "x0 + 1",
                }
            ],
            "default_expression": "round(0.0833*x0**3 - 0.75*x0**2 + 3.1667*x0 - 2)",
        }
    )
    rounded_candidate, _ = E._claude_candidate(
        benchmark, _hypothesis("piecewise_relation", rounded_expression, origin="uncertain")
    )
    assert rounded_candidate is not None
    stored_default = json.loads(rounded_candidate.expression)["default_expression"]
    assert "." not in stored_default
    assert "/" in stored_default
    assert E.predict(rounded_candidate, benchmark, rows) == E.independently_predict(
        rounded_candidate, benchmark, rows
    )


def test_piecewise_undefined_predicate_fails_closed_in_both_evaluators() -> None:
    _, benchmarks = E.load_public_benchmarks(ROOT)
    benchmark = next(item for item in benchmarks if len(item.aliases) == 1)
    expression = json.dumps(
        {
            "branches": [
                {
                    "condition": {"comparator": "lt", "left": "1/x0", "right": "0"},
                    "expression": "-1",
                }
            ],
            "default_expression": "1",
        }
    )
    candidate, _ = E._claude_candidate(
        benchmark, _hypothesis("piecewise_relation", expression, origin="uncertain")
    )
    assert candidate is not None
    rows = (E.Observation((Fraction(0),), Fraction(0)),)
    assert E.predict(candidate, benchmark, rows) == (None,)
    assert E.independently_predict(candidate, benchmark, rows) == (None,)


def test_non_executable_claude_ideas_are_retained_with_origin_and_reason() -> None:
    _, benchmarks = E.load_public_benchmarks(ROOT)
    benchmark = benchmarks[0]
    candidate, record = E._claude_candidate(
        benchmark,
        _hypothesis(
            "other_typed_relation",
            "T_ab = R_ab - R*g_ab/2",
            origin="cross_domain_synthesis",
        ),
    )
    assert candidate is None
    assert record["status"] == "RETAINED_NON_EXECUTABLE"
    assert record["reason"] == "representation_not_yet_executable"
    assert record["llm_self_assessed_origin"] == "cross_domain_synthesis"


def test_malformed_transform_idea_is_retained_instead_of_pruned() -> None:
    _, benchmarks = E.load_public_benchmarks(ROOT)
    candidate, record = E._claude_candidate(
        benchmarks[0],
        _hypothesis(
            "transform_relation",
            "Delta f = f(x + 1) - f(x)",
            origin="uncertain",
        ),
    )
    assert candidate is None
    assert record["status"] == "RETAINED_NON_EXECUTABLE"
    assert record["reason"] == "typed_expression_failed_validation"
    assert record["llm_self_assessed_origin"] == "uncertain"


def test_malformed_piecewise_idea_is_retained_instead_of_pruned() -> None:
    _, benchmarks = E.load_public_benchmarks(ROOT)
    candidate, record = E._claude_candidate(
        benchmarks[0],
        _hypothesis(
            "piecewise_relation",
            json.dumps(
                {
                    "branches": [
                        {
                            "condition": {
                                "comparator": "approximately_lt",
                                "left": "x0",
                                "right": "0",
                            },
                            "expression": "-x0",
                        }
                    ],
                    "default_expression": "x0",
                }
            ),
            origin="uncertain",
        ),
    )
    assert candidate is None
    assert record["status"] == "RETAINED_NON_EXECUTABLE"
    assert record["reason"] == "typed_expression_failed_validation"
    assert record["llm_self_assessed_origin"] == "uncertain"
    assert "comparator" in record["diagnostic"]


def test_malformed_executable_typed_idea_is_retained_instead_of_pruned() -> None:
    _, benchmarks = E.load_public_benchmarks(ROOT)
    candidate, record = E._claude_candidate(
        benchmarks[0],
        _hypothesis(
            "tensor_identity",
            "T_ab = R_ab - R*g_ab/2",
            origin="uncertain",
        ),
    )
    assert candidate is None
    assert record["status"] == "RETAINED_NON_EXECUTABLE"
    assert record["reason"] == "typed_expression_failed_validation"
    assert record["llm_self_assessed_origin"] == "uncertain"


def test_variational_output_cannot_be_decoupled_from_the_checked_claim() -> None:
    _, benchmarks = E.load_public_benchmarks(ROOT)
    specification = {
        "bindings": {"q": "0", "q_ddot": "-x0", "q_dot": "0", "t": "0"},
        "claimed_euler_lagrange": "-q_ddot",
        "coordinate": "t",
        "field": "q",
        "first_derivative": "q_dot",
        "integrand": "q_dot**2/2",
        "output_expression": "x0 + 1",
        "second_derivative": "q_ddot",
    }
    candidate, record = E._claude_candidate(
        next(item for item in benchmarks if len(item.aliases) == 1),
        _hypothesis("variational_principle", json.dumps(specification)),
    )
    assert candidate is None
    assert record["status"] == "RETAINED_NON_EXECUTABLE"
    assert record["reason"] == "typed_expression_failed_validation"
    assert "not induced" in record["diagnostic"]


def test_modular_control_remains_matched_after_canonicalization() -> None:
    public, benchmarks = E.load_public_benchmarks(ROOT)
    benchmark = next(
        item for item in benchmarks if item.benchmark_id == "external.authority-nist-0244"
    )
    target = next(
        item
        for item in E.unseal_targets(ROOT, public, benchmarks)
        if item.benchmark_id == benchmark.benchmark_id
    )
    candidate, _ = E._claude_candidate(
        benchmark,
        _hypothesis(
            "modular_relation",
            '{"expression":"x0*(x0+1)*(2*x0+1)","modulus":6}',
        ),
    )
    assert candidate is not None
    control = E.random_controls(benchmark, {"claude_proposer": (candidate,)}, seed=20260822)[
        "claude_proposer"
    ][0]
    budget = E._load_campaign_config(ROOT, live_claude=False)["search"]["matched_control_budget"]
    assert E._candidate_resource_profile(
        candidate, benchmark, target, budget
    ) == E._candidate_resource_profile(control, benchmark, target, budget)


def test_oversized_executable_candidate_is_retained_without_scoring() -> None:
    public, benchmarks = E.load_public_benchmarks(ROOT)
    benchmark = next(item for item in benchmarks if len(item.aliases) == 1)
    target = next(
        item
        for item in E.unseal_targets(ROOT, public, benchmarks)
        if item.benchmark_id == benchmark.benchmark_id
    )
    expression = json.dumps(
        {
            "branches": [
                {
                    "condition": {
                        "comparator": "lt",
                        "left": "x0" + " + 1" * 10,
                        "right": "0",
                    },
                    "expression": "x0",
                }
            ],
            "default_expression": "x0",
        }
    )
    candidate, record = E._claude_candidate(
        benchmark, _hypothesis("piecewise_relation", expression)
    )
    assert candidate is not None
    assert record["status"] == "ADMITTED_EXECUTABLE"
    budget = E._load_campaign_config(ROOT, live_claude=False)["search"]["matched_control_budget"]
    with pytest.raises(E.ExternalCreativityError, match="exceeds"):
        E._candidate_resource_profile(candidate, benchmark, target, budget)
    result = E._score_candidate(candidate, benchmark, target, budget)
    assert result["scoring_status"] == "RETAINED_UNSCORED_RESOURCE_BUDGET"
    assert result["retention_status"] == "RETAINED_ACTIVE_FOR_REPAIR_OR_LARGER_BUDGET"
    assert "grammar_depth" in result["resource_budget_exceeded"]
    assert "holdout_loss" not in result


def test_piecewise_transform_tensor_and_variational_controls_match_exact_resource_profiles() -> (
    None
):
    public, benchmarks = E.load_public_benchmarks(ROOT)
    benchmark = next(item for item in benchmarks if len(item.aliases) == 1)
    target = next(
        item
        for item in E.unseal_targets(ROOT, public, benchmarks)
        if item.benchmark_id == benchmark.benchmark_id
    )
    tensor = json.dumps(
        {
            "left_components": ["x0", "0", "0", "x0"],
            "output_component": {"flat_index": 0, "side": "left"},
            "right_components": ["x0", "0", "0", "x0"],
            "shape": [2, 2],
            "symmetries": [{"left_axis": 0, "right_axis": 1, "sign": 1}],
            "tensor_name": "T",
            "variance": ["covariant", "covariant"],
        }
    )
    variational = json.dumps(
        {
            "bindings": {"q": "0", "q_ddot": "-x0", "q_dot": "0", "t": "0"},
            "claimed_euler_lagrange": "-q_ddot",
            "coordinate": "t",
            "field": "q",
            "first_derivative": "q_dot",
            "integrand": "q_dot**2/2",
            "second_derivative": "q_ddot",
        }
    )
    transform = json.dumps(
        {
            "claimed_transform": "3*x0**2 + 3*x0 + 1",
            "index": "x0",
            "source_expression": "x0**3",
            "stencil": [
                {"coefficient": "-1", "offset": 0},
                {"coefficient": "1", "offset": 1},
            ],
            "transform_kind": "linear_shift_stencil",
        }
    )
    piecewise = json.dumps(
        {
            "branches": [
                {
                    "condition": {"comparator": "lt", "left": "x0", "right": "3"},
                    "expression": "x0 + 10",
                }
            ],
            "default_expression": "x0 + 20",
        }
    )
    budget = E._load_campaign_config(ROOT, live_claude=False)["search"]["matched_control_budget"]
    for representation, expression in (
        ("piecewise_relation", piecewise),
        ("transform_relation", transform),
        ("tensor_identity", tensor),
        ("variational_principle", variational),
    ):
        candidate, record = E._claude_candidate(benchmark, _hypothesis(representation, expression))
        assert candidate is not None
        assert record["status"] == "ADMITTED_EXECUTABLE"
        control = E.random_controls(
            benchmark,
            {"claude_proposer": (candidate,)},
            seed=20260823,
        )["claude_proposer"][0]
        assert control.expression != candidate.expression
        assert E._candidate_resource_profile(
            candidate, benchmark, target, budget
        ) == E._candidate_resource_profile(control, benchmark, target, budget)


def test_false_transform_tensor_and_variational_claims_execute_but_do_not_score() -> None:
    _, benchmarks = E.load_public_benchmarks(ROOT)
    benchmark = next(item for item in benchmarks if len(item.aliases) == 1)
    rows = tuple(E.Observation((Fraction(index),), Fraction(0)) for index in range(3))
    false_tensor = json.dumps(
        {
            "left_components": ["x0", "0", "0", "x0"],
            "output_component": {"flat_index": 0, "side": "left"},
            "right_components": ["x0 + 1", "0", "0", "x0"],
            "shape": [2, 2],
            "symmetries": [{"left_axis": 0, "right_axis": 1, "sign": 1}],
            "tensor_name": "T",
            "variance": ["covariant", "covariant"],
        }
    )
    false_variational = json.dumps(
        {
            "bindings": {"q": "0", "q_ddot": "x0", "q_dot": "0", "t": "0"},
            "claimed_euler_lagrange": "q_ddot",
            "coordinate": "t",
            "field": "q",
            "first_derivative": "q_dot",
            "integrand": "q_dot**2/2",
            "second_derivative": "q_ddot",
        }
    )
    false_transform = json.dumps(
        {
            "claimed_transform": "3*x0**2 + 3*x0 + 2",
            "index": "x0",
            "source_expression": "x0**3",
            "stencil": [
                {"coefficient": "-1", "offset": 0},
                {"coefficient": "1", "offset": 1},
            ],
            "transform_kind": "linear_shift_stencil",
        }
    )
    for representation, expression in (
        ("transform_relation", false_transform),
        ("tensor_identity", false_tensor),
        ("variational_principle", false_variational),
    ):
        candidate, record = E._claude_candidate(benchmark, _hypothesis(representation, expression))
        assert candidate is not None
        assert record["status"] == "ADMITTED_EXECUTABLE"
        assert E.predict(candidate, benchmark, rows) == (None, None, None)
        assert E.independently_predict(candidate, benchmark, rows) == (
            None,
            None,
            None,
        )


def test_known_and_bounded_unknown_campaign_is_honest(dry_receipt: dict[str, Any]) -> None:
    assert dry_receipt["schema_version"] == E.RECEIPT_SCHEMA
    assert dry_receipt["claims"] == {
        "claude_used_throughout": False,
        "externally_authored_sealed_benchmarks_executed": True,
        "novel_formula_established": False,
        "open_problem_attempted": False,
        "open_problem_solved": False,
    }
    known = [item for item in dry_receipt["benchmarks"] if item["target_kind"] == "known_formula"]
    unknown = [
        item for item in dry_receipt["benchmarks"] if item["target_kind"] == "bounded_unknown"
    ]
    assert len(known) == len(unknown) == 2
    assert all(item["claims"]["known_formula_rediscovered"] for item in known)
    assert all(item["ranked_candidates"][0]["holdout_loss"] == "0" for item in known)
    assert all(not item["claims"]["novel_formula_established"] for item in unknown)
    assert all(item["ranked_candidates"][0]["holdout_loss"] != "0" for item in unknown)
    assert all(not item["bounded_unknown_process_pass"] for item in unknown)


def test_every_family_has_matched_random_and_leave_one_out_controls(
    dry_receipt: dict[str, Any],
) -> None:
    for benchmark in dry_receipt["benchmarks"]:
        metrics = benchmark["family_metrics"]
        ablations = benchmark["family_ablation"]
        assert [item["family"] for item in metrics] == list(E.FAMILY_IDS)
        assert [item["family"] for item in ablations] == list(E.FAMILY_IDS)
        assert all(item["candidate_budget"] == item["matched_random_budget"] for item in metrics)
        assert all(item["candidate_budget"] > 0 for item in metrics)
        assert all(item["candidate_count_match"] for item in metrics)
        assert all(item["grammar_depth_match"] for item in metrics)
        assert all(item["evaluation_runtime_budget_match"] for item in metrics)
        assert all(item["verifier_budget_match"] for item in metrics)
        assert all(item["control_budget_match"] for item in metrics)
        assert all(item["unique_behaviors"] > 0 for item in metrics)
        assert len(benchmark["random_controls"]) == len(E.FAMILY_IDS)
        ranked = {item["candidate_id"]: item for item in benchmark["ranked_candidates"]}
        for controls in benchmark["random_controls"].values():
            for control in controls:
                source = ranked[control["matched_candidate_id"]]
                assert control["resource_profile"] == source["resource_profile"]
        policy = benchmark["matched_control_policy"]
        assert policy["all_family_budgets_match"]
        assert policy["grammar_depth_matched"]
        assert policy["evaluation_runtime_budget_matched"]
        assert policy["verifier_budget_matched"]
        assert policy["deterministic_operation_budget_used"]
        assert not policy["wall_clock_runtime_claimed_matched"]


def test_exact_cas_smt_interval_pass_but_kernel_and_release_fail_closed(
    dry_receipt: dict[str, Any],
) -> None:
    known = [item for item in dry_receipt["benchmarks"] if item["target_kind"] == "known_formula"]
    for benchmark in known:
        assert benchmark["formal_verification"]["backends"] == {
            "cas": True,
            "exact_arithmetic": True,
            "interval": True,
            "lean": False,
            "smt": True,
        }
        assert not benchmark["formal_verification"]["serious_claim_eligible"]
        assert (
            benchmark["prior_art"]["external_literature_index"]["status"]
            == "AUTHORITATIVE_PRIOR_ART_RECORD_FOUND_NOT_NOVELTY_CLEARED"
        )
        assert benchmark["prior_art"]["human_review"]["status"] == "NOT_PERFORMED"
        assert not benchmark["claims"]["serious_claim_released"]
    assert dry_receipt["serious_claim_policy"]["released_claims"] == 0


def test_dataset_pipeline_contains_units_groups_residuals_interventions_and_ood(
    dry_receipt: dict[str, Any],
) -> None:
    for benchmark in dry_receipt["benchmarks"]:
        evidence = benchmark["dataset_evidence"]
        assert evidence["dimension_basis"] == ["mass", "length", "time"]
        assert evidence["dimension_matrix_rank"] >= 0
        assert "dimension_solution_set" in evidence
        assert "dimensionless_group_basis" in evidence
        assert evidence["unit_normalization"]["inputs"]
        assert evidence["symmetry_groups"]["declared_coordinates"]
        assert evidence["residual_channels"]["declared"]
        assert evidence["causal_interventions"]["declared"]
        assert (
            evidence["causal_interventions"]["execution_status"]
            == "DECLARED_REQUIRES_INTERVENTION_DATA"
        )
        assert not evidence["causal_interventions"][
            "observational_rows_mislabelled_as_interventions"
        ]
        assert evidence["ood_split_rule"]
        assert evidence["holdout_opened_last"]


def test_proof_plan_search_and_independent_exact_implementation_are_recorded(
    dry_receipt: dict[str, Any],
) -> None:
    for benchmark in dry_receipt["benchmarks"]:
        proof_search = benchmark["proof_plan_search"]
        assert proof_search["selected_route"][0] == "exact_row_replay"
        plan_names = {item["plan"] for item in proof_search["plans"]}
        assert {
            "bijection_construction",
            "contradiction_via_modular_obstruction",
            "induction_on_recurrence",
            "invariant_strengthening",
            "lean_kernel_bridge",
            "minimal_counterexample_descent",
            "transform_domain_identity",
        } <= plan_names
        assert len(proof_search["selected_route"]) <= 5
        reproduction = benchmark["independent_exact_reproduction"]
        assert reproduction["implementation"] == "python_stdlib_fraction_ast_v1"
        assert reproduction["match"]
        assert not reproduction["shared_symbolic_runtime"]


def test_target_is_opened_after_claude_and_proposal_seals(dry_receipt: dict[str, Any]) -> None:
    events = dry_receipt["blind_chronology"]
    assert [item["sequence"] for item in events] == list(range(len(events)))
    target_open = next(item for item in events if item["target_reads"] == 1)
    proposal_seal = next(
        item for item in events if item["event"] == "proposal_roots_and_train_evidence_sealed"
    )
    critique = next(
        item for item in events if item["event"] == "claude_blind_critique_completed_or_blocked"
    )
    control_seal = next(
        item for item in events if item["event"] == "matched_random_controls_sealed"
    )
    assert (
        proposal_seal["sequence"]
        < critique["sequence"]
        < control_seal["sequence"]
        < target_open["sequence"]
    )
    assert all(item["target_reads"] == 0 for item in events[: target_open["sequence"]])


def test_live_claude_fixture_proposes_and_steers_without_verifying(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-secret")
    transport = CampaignClaudeTransport()
    journal_path = tmp_path / "live-call-attempts.jsonl"
    receipt = E.run_campaign(
        ROOT,
        live_claude=True,
        claude_transport=transport,
        attempt_journal_path=journal_path,
    )
    assert receipt["claude"]["status"] == "PASS"
    assert receipt["claude"]["completed_calls"] == receipt["claude"]["required_calls"] == 8
    assert receipt["claude"]["proposer_hypotheses"] == 4
    assert receipt["claude"]["steering_actions"] == 4
    assert receipt["claims"]["claude_used_throughout"]
    for item in receipt["benchmarks"]:
        admission = item["proposer_admission"]
        assert admission["admitted_executable_hypotheses"] == 1
        assert admission["non_executable_typed_hypotheses"] == 0
        assert admission["proposed_hypotheses"] == 1
        assert len(admission["records"]) == 1
        assert admission["records"][0]["status"] == "ADMITTED_EXECUTABLE"
        contribution = item["claude_contribution"]
        assert contribution["status"] == "MEASURED_EXECUTABLE_CLAUDE_CONTRIBUTION"
        assert contribution["retained_unscored_executable_candidates"] == 0
        assert contribution["scored_executable_candidates"] == 1
        assert contribution["grammar_depth_match"]
        assert contribution["evaluation_runtime_budget_match"]
        assert contribution["verifier_budget_match"]
        assert len(item["claude_matched_controls"]) == 1
    assert receipt["serious_claim_policy"]["released_claims"] == 0
    assert [item["method"] for item in transport.requests].count("GET") == 1
    assert [item["method"] for item in transport.requests].count("POST") == 8
    proposer_prompts = [
        json.loads(item["body"]["messages"][0]["content"])
        for item in transport.requests
        if item["method"] == "POST"
        and json.loads(item["body"]["messages"][0]["content"])["role"] == "proposer"
    ]
    assert len(proposer_prompts) == 4
    assert all(
        prompt["instruction"] == E.EXECUTABLE_PROPOSER_INSTRUCTION for prompt in proposer_prompts
    )
    assert "transform_relation uses JSON" in E.EXECUTABLE_PROPOSER_INSTRUCTION
    assert "piecewise_relation uses JSON" in E.EXECUTABLE_PROPOSER_INSTRUCTION
    assert "tensor_identity uses JSON" in E.EXECUTABLE_PROPOSER_INSTRUCTION
    assert "variational_principle uses JSON" in E.EXECUTABLE_PROPOSER_INSTRUCTION
    assert "fixture-secret" not in json.dumps(receipt, sort_keys=True)
    journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
    assert [row["kind"] for row in journal].count("attempt_started") == 1
    assert [row["kind"] for row in journal].count("claude_call_completed_or_blocked") == 8
    assert [row["kind"] for row in journal].count("attempt_completed") == 1
    assert len({row["attempt_id"] for row in journal}) == 1
    assert "fixture-secret" not in json.dumps(journal, sort_keys=True)


def test_live_campaign_retains_oversized_claude_branches_instead_of_aborting(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-secret")
    expression = json.dumps(
        {
            "branches": [
                {
                    "condition": {
                        "comparator": "lt",
                        "left": "x0" + " + 1" * 10,
                        "right": "0",
                    },
                    "expression": "x0",
                }
            ],
            "default_expression": "x0",
        }
    )
    transport = CampaignClaudeTransport(expression, "piecewise_relation")
    receipt = E.run_campaign(
        ROOT,
        live_claude=True,
        claude_transport=transport,
        attempt_journal_path=tmp_path / "overflow-attempts.jsonl",
    )
    assert receipt["claude"]["completed_calls"] == 8
    for benchmark in receipt["benchmarks"]:
        retained = benchmark["retained_unscored_candidates"]
        assert len(retained) == 1
        assert retained[0]["family"] == "claude_proposer"
        assert retained[0]["scoring_status"] == "RETAINED_UNSCORED_RESOURCE_BUDGET"
        contribution = benchmark["claude_contribution"]
        assert contribution["status"] == ("RETAINED_UNSCORED_OR_NON_EXECUTABLE_CLAUDE_PROPOSALS")
        assert contribution["retained_unscored_executable_candidates"] == 1
        assert contribution["scored_executable_candidates"] == 0
        assert benchmark["claude_matched_controls"] == []


def test_open_problem_spend_and_reproduction_remain_blocked(dry_receipt: dict[str, Any]) -> None:
    gate = dry_receipt["open_problem_gate"]
    assert gate["level5_process_passes"] == 0
    assert gate["minimum_independent_level5_passes"] == 3
    assert not gate["authorized"]
    assert gate["public_failure_receipt_required"]
    assert gate["success_criteria"]["sealed_holdout_loss"] == "0"
    reproduction = dry_receipt["independent_reproduction"]
    assert reproduction["status"] == "IMPLEMENTATIONS_PASS_MACHINE_PENDING"
    assert reproduction["received_machines"] < reproduction["minimum_machines"]
    assert reproduction["received_implementations"] == reproduction["minimum_implementations"]


def test_dry_receipt_replays_byte_for_byte(dry_receipt: dict[str, Any]) -> None:
    assert E.run_campaign(ROOT) == dry_receipt
    assert json.loads((ROOT / E.OUTPUT_PATH).read_text(encoding="utf-8")) == dry_receipt
    assert set(dry_receipt["config"]) == {
        "campaign_sha256",
        "claude_source_sha256",
        "claude_transport_source_sha256",
        "independent_evaluator_sha256",
        "lean_source_sha256",
        "public_benchmarks_sha256",
        "sealed_targets_sha256",
        "source_sha256",
        "test_sha256",
    }
    assert dry_receipt["content_sha256"] == E.canonical_sha256(
        {key: value for key, value in dry_receipt.items() if key != "content_sha256"}
    )
