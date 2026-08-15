from __future__ import annotations

import copy
from fractions import Fraction

import pytest
import sympy as sp

from sigma_theory_compiler.math_types import RATIONAL, REAL
from sigma_theory_compiler.sigma_core import DomainPackRef, ProvenanceRecord, canonical_sha256
from sigma_theory_compiler.symbolic_candidate_generator import (
    SCOPE,
    SymbolicCandidateGenerator,
    SymbolicGenerationBatch,
    SymbolicGeneratorBudget,
    SymbolicGeneratorError,
    SymbolicTemplate,
    SymbolicVariable,
)

PACK = DomainPackRef("math.symbolic", "1.0", "2" * 64)
BUDGET = SymbolicGeneratorBudget(
    max_templates=4,
    max_coefficients_per_axis=4,
    max_candidates=32,
    max_work_items=64,
)
X = sp.Symbol("x")
A = sp.Symbol("a")
B = sp.Symbol("b")


def _template(template_id: str, expression: sp.Expr | None = None) -> SymbolicTemplate:
    return SymbolicTemplate.create(
        template_id,
        A * X + B if expression is None else expression,
        variables=(SymbolicVariable("x", RATIONAL),),
        coefficient_symbols=("a", "b"),
    )


def _batch() -> SymbolicGenerationBatch:
    return SymbolicCandidateGenerator.generate(
        (_template("affine_a"), _template("affine_b", B + X * A)),
        (-1, 0, 1),
        domain_pack=PACK,
        budget=BUDGET,
    )


def test_exact_bounded_generation_canonical_dedup_and_typed_sigma_artifacts() -> None:
    batch = _batch()

    assert batch.generated_before_deduplication == 18
    assert len(batch.candidates) == 9
    assert batch.duplicates_removed == 9
    assert sum(item.disposition == "representative" for item in batch.origins) == 9
    assert sum(item.disposition == "deduplicated_equivalent" for item in batch.origins) == 9
    assert all(isinstance(item.provenance, ProvenanceRecord) for item in batch.candidates)
    assert all(
        item.representation["typed_variables"][0]["math_type"] == repr(RATIONAL)
        for item in batch.candidates
    )
    assert all(item.claims == ("generated_candidate",) for item in batch.candidates)
    assert batch.scope == SCOPE
    serialized = batch.to_dict()
    assert "truth" not in serialized
    assert "promotion" not in serialized


def test_exact_fraction_coefficients_and_safe_functions_are_supported() -> None:
    sine = SymbolicTemplate.create(
        "safe_sine",
        A * sp.sin(X),
        variables=(SymbolicVariable("x", RATIONAL),),
        coefficient_symbols=("a",),
        allowed_functions=("sin",),
    )
    batch = SymbolicCandidateGenerator.generate(
        (sine,),
        (Fraction(-1, 2), Fraction(1, 3)),
        domain_pack=PACK,
        budget=BUDGET,
    )

    assert batch.generated_before_deduplication == 2
    assert len(batch.candidates) == 2
    assert [item.fraction for item in batch.coefficient_values] == [
        Fraction(-1, 2),
        Fraction(1, 3),
    ]


def test_generation_and_closed_receipt_replay_are_deterministic() -> None:
    first = _batch()
    second = _batch()

    assert first == second
    assert first.content_sha256 == second.content_sha256
    assert first.lineage_sha256 == second.lineage_sha256
    assert SymbolicGenerationBatch.from_dict(first.to_dict()) == first


@pytest.mark.parametrize(
    ("expression", "variables", "coefficients", "functions", "message"),
    [
        (A * X + sp.Float("0.5"), (SymbolicVariable("x", RATIONAL),), ("a",), (), "floating-point"),
        (A * X + sp.Symbol("y"), (SymbolicVariable("x", RATIONAL),), ("a",), (), "unknown symbols"),
        (A * sp.gamma(X), (SymbolicVariable("x", RATIONAL),), ("a",), (), "unsafe functions"),
        (A * X, (SymbolicVariable("x", RATIONAL),), ("a", "b"), (), "omits coefficient"),
        (X**A, (SymbolicVariable("x", RATIONAL),), ("a",), (), "symbolic exponents"),
    ],
)
def test_float_unknown_symbol_unsafe_function_and_unbounded_templates_fail_closed(
    expression: sp.Expr,
    variables: tuple[SymbolicVariable, ...],
    coefficients: tuple[str, ...],
    functions: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(SymbolicGeneratorError, match=message):
        SymbolicTemplate.create(
            "invalid_template",
            expression,
            variables=variables,
            coefficient_symbols=coefficients,
            allowed_functions=functions,
        )


def test_unsupported_variable_type_and_float_coefficient_fail_closed() -> None:
    with pytest.raises(SymbolicGeneratorError, match="IntegerType or RationalType"):
        SymbolicVariable("x", REAL)

    with pytest.raises(SymbolicGeneratorError, match="exact int or Fraction"):
        SymbolicCandidateGenerator.generate(
            (_template("affine"),),
            (0.5,),
            domain_pack=PACK,
            budget=BUDGET,
        )


def test_template_coefficient_candidate_and_work_budgets_fail_closed() -> None:
    with pytest.raises(SymbolicGeneratorError, match="max_templates"):
        SymbolicCandidateGenerator.generate(
            (_template("one"), _template("two")),
            (0, 1),
            domain_pack=PACK,
            budget=SymbolicGeneratorBudget(1, 4, 32, 64),
        )
    with pytest.raises(SymbolicGeneratorError, match="max_coefficients_per_axis"):
        SymbolicCandidateGenerator.generate(
            (_template("one"),),
            (0, 1, 2),
            domain_pack=PACK,
            budget=SymbolicGeneratorBudget(1, 2, 32, 64),
        )
    with pytest.raises(SymbolicGeneratorError, match="max_work_items"):
        SymbolicCandidateGenerator.generate(
            (_template("one"),),
            (-1, 0, 1),
            domain_pack=PACK,
            budget=SymbolicGeneratorBudget(1, 3, 32, 8),
        )
    with pytest.raises(SymbolicGeneratorError, match="max_candidates"):
        SymbolicCandidateGenerator.generate(
            (_template("one"),),
            (-1, 0, 1),
            domain_pack=PACK,
            budget=SymbolicGeneratorBudget(1, 3, 2, 16),
        )


def test_nested_candidate_and_receipt_hash_tamper_fail_closed() -> None:
    batch = _batch()
    candidate_tamper = copy.deepcopy(batch.to_dict())
    candidate_tamper["candidates"][0]["representation"]["origin_count"] = 999
    with pytest.raises(SymbolicGeneratorError, match="Sigma Core binding"):
        SymbolicGenerationBatch.from_dict(candidate_tamper)

    origin_tamper = copy.deepcopy(batch.to_dict())
    origin_tamper["origins"][0]["disposition"] = "deduplicated_equivalent"
    with pytest.raises(SymbolicGeneratorError, match="origin canonical hash"):
        SymbolicGenerationBatch.from_dict(origin_tamper)

    unknown = copy.deepcopy(batch.to_dict())
    unknown["proof_status"] = "pass"
    with pytest.raises(SymbolicGeneratorError, match="keys changed"):
        SymbolicGenerationBatch.from_dict(unknown)


def test_resealed_scope_and_disposition_semantic_tamper_fail_closed() -> None:
    batch = _batch()
    scope_tamper = copy.deepcopy(batch.to_dict())
    scope_tamper["scope"] = "canonicalization proves truth"
    body = {key: value for key, value in scope_tamper.items() if key != "content_sha256"}
    scope_tamper["content_sha256"] = canonical_sha256(body)
    with pytest.raises(SymbolicGeneratorError, match="schema or scope"):
        SymbolicGenerationBatch.from_dict(scope_tamper)

    disposition_tamper = copy.deepcopy(batch.to_dict())
    origin = disposition_tamper["origins"][0]
    origin["disposition"] = "deduplicated_equivalent"
    origin_body = {key: value for key, value in origin.items() if key != "content_sha256"}
    origin["content_sha256"] = canonical_sha256(origin_body)
    disposition_tamper["lineage_sha256"] = canonical_sha256(
        {
            "schema_version": disposition_tamper["schema_version"],
            "domain_pack": disposition_tamper["domain_pack"],
            "budget": disposition_tamper["budget"],
            "template_receipts": [
                item["content_sha256"] for item in disposition_tamper["template_descriptors"]
            ],
            "coefficient_values": disposition_tamper["coefficient_values"],
            "candidate_refs": [
                {
                    "artifact_id": item["artifact_id"],
                    "content_sha256": item["content_sha256"],
                }
                for item in disposition_tamper["candidates"]
            ],
            "origin_receipts": [item["content_sha256"] for item in disposition_tamper["origins"]],
        }
    )
    batch_body = {
        key: value for key, value in disposition_tamper.items() if key != "content_sha256"
    }
    disposition_tamper["content_sha256"] = canonical_sha256(batch_body)
    with pytest.raises(SymbolicGeneratorError, match="disposition does not match"):
        SymbolicGenerationBatch.from_dict(disposition_tamper)
