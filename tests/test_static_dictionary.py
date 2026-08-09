from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.action_ir import compile_action_file
from sigma_theory_compiler.cli import _parser
from sigma_theory_compiler.static_dictionary import (
    audit_priority_static_lift,
    classify_generator_expression,
    compile_static_dictionary_ir,
)

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs" / "covariant_action_grammar.json"
CONTRACT = ROOT / "configs" / "covariant_field_contract.json"


def _dictionary(name: str = "einstein_aether_control.json") -> dict:
    action = compile_action_file(ROOT / "configs" / "actions" / name, GRAMMAR, CONTRACT)
    return compile_static_dictionary_ir(action)


def test_static_aether_tensor_reduction_is_exact_and_hash_bound() -> None:
    action = compile_action_file(
        ROOT / "configs" / "actions" / "einstein_aether_control.json",
        GRAMMAR,
        CONTRACT,
    )
    result = compile_static_dictionary_ir(action)
    assert result == compile_static_dictionary_ir(action)
    assert result["status"] == "pass"
    assert result["input_action_sha256"] == action["content_sha256"]
    aether = result["static_reductions"]["aether"]
    assert aether["orthogonality_residual"] == "0"
    assert aether["K1_over_Lu2"] == "-a1**2 - a2**2 - a3**2"
    assert aether["K2_over_Lu2"] == "0"
    assert aether["K3_over_Lu2"] == "0"
    assert aether["K4_over_Lu2"] == "a1**2 + a2**2 + a3**2"
    assert (
        result["aether_static_acceleration_sector"][
            "action_density_coefficient_of_a_i_a^i"
        ]
        == "M_Pl**2*(c1 + c4)/2"
    )


def test_scalar_proca_eh_and_baryonic_diagnostic_reductions_are_exact() -> None:
    reductions = _dictionary()["static_reductions"]
    assert reductions["scalar"]["X_phi"] == reductions["scalar"]["expected"]
    assert (
        reductions["proca"]["F_mu_nu_F_mu_nu"]
        == reductions["proca"]["expected"]
    )
    assert reductions["einstein_hilbert"]["density_split_residual"] == "0"
    baryon = reductions["baryonic_diagnostic"]
    assert baryon["z_b"] == "n_b**2/n_0**2"
    assert baryon["generator_status"] == "forbidden action atom; diagnostic only"


def test_generator_expression_lift_decisions_fail_closed() -> None:
    forbidden = classify_generator_expression("q-z", aether_x_available=True)
    assert forbidden["decision"] == "reject_forbidden_baryonic_action_atom"
    missing_q = classify_generator_expression("q+q**2", aether_x_available=True)
    assert missing_q["decision"] == "unresolved_missing_covariant_q_atom"
    linear_x = classify_generator_expression("2*x", aether_x_available=True)
    assert linear_x["decision"] == "supported_linear_aether_x_lift"
    nonlinear_x = classify_generator_expression("sqrt(1+x)-1", aether_x_available=True)
    assert nonlinear_x["decision"] == (
        "unresolved_missing_nonlinear_aether_acceleration_adapter"
    )
    without_aether = classify_generator_expression("x", aether_x_available=False)
    assert without_aether["decision"] == "unresolved_no_unit_aether_x_dictionary"


def test_generator_expression_parser_rejects_executable_or_unknown_syntax() -> None:
    with pytest.raises(TypeError, match="unsupported generator expression syntax"):
        classify_generator_expression("__import__('os').system('echo unsafe')", aether_x_available=True)
    with pytest.raises(TypeError, match="unsupported generator expression syntax"):
        classify_generator_expression("unknown+x", aether_x_available=True)


def test_dense_priority_queue_has_no_current_covariant_lift() -> None:
    priority = json.loads(
        (ROOT / "runs" / "knowledge-base" / "generated-priority-dense.json").read_text(
            encoding="utf-8"
        )
    )
    result = audit_priority_static_lift(priority, _dictionary())
    assert result["queue_count"] == 124
    assert result["currently_liftable_count"] == 0
    assert result["q_backend_queue_count"] == 20
    assert result["decision_counts"] == {
        "reject_forbidden_baryonic_action_atom": 104,
        "unresolved_missing_covariant_q_atom": 20,
    }
    assert not result["observational_data_opened"]
    assert "Aether acceleration-gradient q invariant" in result["next_backend_target"]


def test_static_dictionary_marks_x_unavailable_without_unit_aether() -> None:
    scalar = _dictionary("canonical_scalar_control.json")
    assert scalar["status"] == "pass"
    assert (
        scalar["legacy_generator_dictionary"]["x"]["status"]
        == "unavailable_for_this_action"
    )


def test_static_dictionary_cli_contracts_are_explicit() -> None:
    derived = _parser().parse_args(
        [
            "action-static-dictionary",
            "--spec",
            "action.json",
            "--generator-expression",
            "q+x",
            "--output",
            "dictionary.json",
        ]
    )
    assert derived.command == "action-static-dictionary"
    assert derived.generator_expression == "q+x"
    audit = _parser().parse_args(
        [
            "static-lift-audit",
            "--dictionary",
            "dictionary.json",
            "--priority",
            "priority.json",
            "--output",
            "audit.json",
        ]
    )
    assert audit.command == "static-lift-audit"


def test_generated_q_q2_candidate_has_exact_static_shape_but_not_formal_health() -> None:
    action = compile_action_file(
        ROOT
        / "configs"
        / "actions"
        / "generated_gf_cb4ebf3da5a74582_q_q2_candidate.json",
        GRAMMAR,
        CONTRACT,
    )
    result = compile_static_dictionary_ir(action)
    q_dictionary = result["legacy_generator_dictionary"]["q"]
    assert result["status"] == "pass"
    assert q_dictionary["status"] == "derived_and_generator_matched"
    assert q_dictionary["normalized_action_shape"] == "q*(q + 1)"
    assert q_dictionary["exact_shape_match"] is True
    assert result["generator_expression_classification"]["decision"] == (
        "supported_exact_projected_aether_q_lift"
    )
    reduction = result["static_reductions"]["aether_acceleration_gradient"]
    assert reduction["projected_contraction"] == reduction["expected_spatial_contraction"]
    assert "tilted u" in reduction["generic_tilt_warning"]


def test_mixed_q_sqrt_x_candidate_has_exact_static_shape() -> None:
    action = compile_action_file(
        ROOT
        / "configs"
        / "actions"
        / "generated_gf_5df8715b319f54cb_q_sqrtx_candidate.json",
        GRAMMAR,
        CONTRACT,
    )
    result = compile_static_dictionary_ir(action)
    assert action["valid"], action["errors"]
    assert result["status"] == "pass"
    assert result["legacy_generator_dictionary"]["q"]["exact_shape_match"] is True
    assert result["legacy_generator_dictionary"]["q"]["normalized_action_shape"] == (
        "q + sqrt(x + 1) - 1"
    )
    assert result["legacy_generator_dictionary"]["x"]["status"] == (
        "derived_and_generator_matched"
    )


def test_static_null_completion_preserves_the_exact_generator_shape() -> None:
    action = compile_action_file(
        ROOT
        / "configs"
        / "actions"
        / "generated_gf_5df8715b319f54cb_static_null_completion.json",
        GRAMMAR,
        CONTRACT,
    )
    result = compile_static_dictionary_ir(action)
    assert action["valid"], action["errors"]
    assert result["status"] == "pass"
    assert result["legacy_generator_dictionary"]["q"]["exact_shape_match"] is True
    assert result["aether_static_acceleration_sector"][
        "action_density_coefficient_of_a_i_a^i"
    ] == "0"
    assert action["canonical"]["covariant_completion"]["static_null_terms"] == [
        "AETHER_K1",
        "AETHER_K4",
    ]
