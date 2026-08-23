from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler import expanded_math_grammar_controls as C
from sigma_theory_compiler import expanded_math_independent_evaluator as independent
from sigma_theory_compiler import expanded_math_primary_evaluator as primary
from sigma_theory_compiler.math_expression_ir import (
    ExpressionIRError,
    FiniteProduct,
    FiniteSum,
    GeneratingFunction,
    ModularRelation,
    TensorIdentity,
    VariationalFunctional,
    add,
    formula_to_data,
    literal,
    multiply,
    power,
    symbol,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _examples():
    index = symbol("i")
    variable = symbol("x")
    field = symbol("q")
    first = symbol("q_dot")
    second = symbol("q_ddot")
    return (
        (FiniteSum(index, 1, 5, index, literal(15)), {}),
        (FiniteProduct(index, 1, 4, index, literal(24)), {}),
        (
            GeneratingFunction(
                "geometric",
                variable,
                tuple(literal(1) for _ in range(5)),
                add(*(power(variable, exponent) for exponent in range(5))),
            ),
            {"x": Fraction(1, 2)},
        ),
        (ModularRelation(power(2, 5), literal(1), 31), {}),
        (
            TensorIdentity(
                "symmetric",
                (2, 2),
                ("covariant", "covariant"),
                tuple(literal(value) for value in (1, 2, 2, 3)),
                tuple(literal(value) for value in (1, 2, 2, 3)),
                ((0, 1, 1),),
            ),
            {},
        ),
        (
            VariationalFunctional(
                "q",
                "t",
                "q_dot",
                "q_ddot",
                add(
                    multiply(Fraction(1, 2), power(first, 2)),
                    multiply(-2, power(field, 2)),
                ),
                add(multiply(-4, field), -second),
            ),
            {},
        ),
    )


def test_expanded_nodes_are_first_class_serializable_formulas() -> None:
    assert [formula_to_data(formula)["kind"] for formula, _ in _examples()] == [
        "finite_sum",
        "finite_product",
        "generating_function",
        "modular_relation",
        "tensor_identity",
        "variational_functional",
    ]


def test_primary_and_independent_evaluators_agree_on_every_expanded_kind() -> None:
    for formula, assignment in _examples():
        assert primary.evaluate(formula, assignment) is True
        assert independent.evaluate(formula, assignment) is True


def test_mutations_are_rejected_by_both_evaluators() -> None:
    index = symbol("i")
    wrong_sum = FiniteSum(index, 1, 5, index, literal(14))
    wrong_tensor = TensorIdentity(
        "symmetric",
        (2, 2),
        ("covariant", "covariant"),
        tuple(literal(value) for value in (1, 2, 2, 3)),
        tuple(literal(value) for value in (1, 2, 4, 3)),
        ((0, 1, 1),),
    )
    for formula in (wrong_sum, wrong_tensor):
        assert primary.evaluate(formula) is False
        assert independent.evaluate(formula) is False


def test_resource_and_tensor_type_boundaries_fail_closed() -> None:
    index = symbol("i")
    with pytest.raises(ExpressionIRError, match="finite-term budget"):
        FiniteSum(index, 0, 64, index, literal(0))
    with pytest.raises(ExpressionIRError, match="component budget"):
        TensorIdentity(
            "too_large",
            (8, 8, 8),
            ("covariant",) * 3,
            tuple(literal(0) for _ in range(512)),
            tuple(literal(0) for _ in range(512)),
        )
    with pytest.raises(ExpressionIRError, match="symmetry"):
        TensorIdentity(
            "bad_symmetry",
            (2, 3),
            ("covariant", "covariant"),
            tuple(literal(0) for _ in range(6)),
            tuple(literal(0) for _ in range(6)),
            ((0, 1, 1),),
        )
    with pytest.raises(ExpressionIRError, match="symmetry"):
        TensorIdentity(
            "boolean_symmetry",
            (2, 2),
            ("covariant", "covariant"),
            tuple(literal(0) for _ in range(4)),
            tuple(literal(0) for _ in range(4)),
            ((0, 1, True),),
        )
    with pytest.raises(ExpressionIRError, match="bound index"):
        FiniteSum(index, 0, 3, index, index + 1)
    with pytest.raises(ExpressionIRError, match="series variable"):
        GeneratingFunction("bad", symbol("x"), (symbol("x"),), literal(0))
    with pytest.raises(ExpressionIRError, match="second derivative"):
        VariationalFunctional(
            "q",
            "t",
            "q_dot",
            "q_ddot",
            symbol("q_ddot"),
            literal(0),
        )


def test_independent_evaluator_source_does_not_import_sympy() -> None:
    for path in (C.INDEPENDENT_PATH, C.INDEPENDENT_BASE_PATH):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "import sympy" not in source
        assert "from sympy" not in source


def test_inexact_float_assignments_fail_closed_in_both_evaluators() -> None:
    formula = GeneratingFunction(
        "constant",
        symbol("x"),
        (literal(1),),
        literal(1),
    )
    with pytest.raises(primary.ExpandedPrimaryEvaluationError, match="assignment"):
        primary.evaluate(formula, {"x": 0.5})
    with pytest.raises(independent.ExpandedIndependentEvaluationError, match="assignment"):
        independent.evaluate(formula, {"x": 0.5})


def test_control_receipt_covers_every_kind_with_positive_and_mutation_evidence() -> None:
    receipt = C.build_receipt(ROOT)
    C.validate_receipt(receipt, ROOT)
    assert receipt["summary"] == {
        "admitted_formula_kinds": list(C._KINDS),
        "controls_passed": 7,
        "controls_total": 7,
        "status": "PASS_EXPANDED_TYPED_GRAMMAR_CONTROLS",
    }
    assert all(row["primary_positive_passed"] for row in receipt["controls"])
    assert all(row["independent_positive_passed"] for row in receipt["controls"])
    assert all(row["primary_mutation_rejected"] for row in receipt["controls"])
    assert all(row["independent_mutation_rejected"] for row in receipt["controls"])


def test_stored_control_receipt_validates_against_current_sources() -> None:
    receipt = json.loads((ROOT / C.OUTPUT_PATH).read_text(encoding="utf-8"))
    C.validate_receipt(receipt, ROOT)


def test_resealed_receipt_cannot_promote_grammar_controls_to_novelty() -> None:
    receipt = C.build_receipt(ROOT)
    changed = copy.deepcopy(receipt)
    changed["claims"]["control_pass_establishes_literature_novelty"] = True
    changed["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(C.ExpandedGrammarControlError, match="policy"):
        C.validate_receipt(changed)


def test_resealed_receipt_cannot_rebind_an_evaluator_source_path() -> None:
    receipt = C.build_receipt(ROOT)
    changed = copy.deepcopy(receipt)
    changed["source_bindings"]["primary_evaluator"] = dict(
        changed["source_bindings"]["math_ir"]
    )
    changed["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(C.ExpandedGrammarControlError, match="source path"):
        C.validate_receipt(changed, ROOT)
